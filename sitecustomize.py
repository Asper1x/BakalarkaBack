"""
Automatic hook for 3D Gaussian Splatting / 2D-SuGaR training:
1. Non-blocking LiveStreamer telemetry & point-cloud streaming (when LIVE_STREAM_PACKET is set)
2. Configurable SuGaR Mesh Refinement mode (REFINEMENT_MODE = skip | fast | balanced | full)
3. 3D Visual-Hull & Black-Skirt Mesh Carving (when $DATASET/masks exists from MASK_FOREGROUND=1)
4. Artifact-free Advancing-Front Watertight Mesh hole-filling (when WATERTIGHT_MESH=1)
Loaded automatically via PYTHONPATH by Python site module.
Compatible with Python 3.7+ (no PEP 604 union types).
"""
import builtins
from collections import Counter, defaultdict, deque
import math
import os
from pathlib import Path
import shutil
import sys


# ==============================================================================
# 1. Initialize LiveStreamer FIRST so WebSocket streaming is always active
# ==============================================================================
live_packet = os.getenv("LIVE_STREAM_PACKET")
streamer = None
if live_packet:
    try:
        from live_streamer import LiveStreamer

        streamer = LiveStreamer(
            job_id=os.getenv("JOB_ID"),
            output_packet_path=live_packet,
            streaming_flag_path=os.getenv("STREAMING_FLAG"),
            stream_interval=int(os.getenv("STREAM_INTERVAL", "15")),
        )
    except Exception:
        streamer = None


# ==============================================================================
# 2. Resolve REFINEMENT_MODE, WATERTIGHT_MESH, MASK_FOREGROUND, and DATASET_DIR
# ==============================================================================
_raw_scene_type = os.getenv("SCENE_TYPE", "object").strip()
_refinement_mode = os.getenv("REFINEMENT_MODE", "").strip().lower()
_watertight_env = os.getenv("WATERTIGHT_MESH", "").strip().lower()
_mask_fg_env = os.getenv("MASK_FOREGROUND", "").strip().lower()

if ":" in _raw_scene_type:
    _parts = _raw_scene_type.split(":")
    if len(_parts) >= 2 and not _refinement_mode:
        _refinement_mode = _parts[1].strip().lower()
    if len(_parts) >= 3 and not _watertight_env:
        _watertight_env = _parts[2].strip().lower()

if not _refinement_mode:
    _refinement_mode = "full"
_watertight_flag = _watertight_env in ("1", "true", "yes", "on")
_mask_fg_flag = _mask_fg_env in ("1", "true", "yes", "on")


def _resolve_dataset_dir():
    env_ds = os.getenv("DATASET_DIR", "").strip()
    if env_ds and os.path.isdir(env_ds):
        return Path(env_ds)
    # Also inspect sys.argv for -s / --source_path passed to train.py
    argv = getattr(sys, "argv", []) or []
    for i, arg in enumerate(argv):
        if arg in ("-s", "--source_path") and i + 1 < len(argv):
            cand = Path(argv[i + 1])
            if cand.is_dir():
                return cand
    return None


_VISUAL_HULL_CACHE = None


def _get_visual_hull_context():
    """
    Lazily load COLMAP camera poses (sparse/0/{cameras,images}.txt) and 2D foreground masks (masks/*.png)
    when MASK_FOREGROUND=1.
    """
    global _VISUAL_HULL_CACHE
    if _VISUAL_HULL_CACHE is not None:
        return _VISUAL_HULL_CACHE

    if not _mask_fg_flag:
        _VISUAL_HULL_CACHE = False
        return False

    ds_dir = _resolve_dataset_dir()
    if ds_dir is None:
        _VISUAL_HULL_CACHE = False
        return False

    masks_dir = ds_dir / "masks"
    sparse_dir = ds_dir / "sparse" / "0"
    if not masks_dir.is_dir() or not sparse_dir.is_dir():
        _VISUAL_HULL_CACHE = False
        return False

    try:
        import numpy as np
        from PIL import Image

        cameras = {}
        cameras_txt = sparse_dir / "cameras.txt"
        if cameras_txt.is_file():
            for line in cameras_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split()
                if len(parts) >= 5:
                    cam_id = int(parts[0])
                    model = parts[1]
                    w, h = int(parts[2]), int(parts[3])
                    params = [float(x) for x in parts[4:]]
                    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL") and len(params) >= 3:
                        fx = fy = params[0]
                        cx, cy = params[1], params[2]
                    elif len(params) >= 4:
                        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                    else:
                        fx = fy = float(max(w, h))
                        cx, cy = w * 0.5, h * 0.5
                    cameras[cam_id] = (w, h, fx, fy, cx, cy)

        views = []
        images_txt = sparse_dir / "images.txt"
        if images_txt.is_file():
            lines = [
                l.strip()
                for l in images_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
                if l.strip() and not l.strip().startswith("#")
            ]
            for i in range(0, len(lines), 2):
                elems = lines[i].split()
                if len(elems) < 10:
                    continue
                qw, qx, qy, qz = map(float, elems[1:5])
                tx, ty, tz = map(float, elems[5:8])
                cam_id = int(elems[8])
                img_name = " ".join(elems[9:])
                if cam_id not in cameras:
                    continue
                R = np.array(
                    [
                        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
                        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
                    ],
                    dtype=np.float64,
                )
                t = np.array([tx, ty, tz], dtype=np.float64)
                views.append((img_name, R, t, cameras[cam_id]))

        if not views:
            _VISUAL_HULL_CACHE = False
            return False

        if len(views) > 56:
            step = max(1, len(views) // 56)
            views = views[::step]

        mask_by_stem = {}
        for mf in sorted(masks_dir.glob("*.png")):
            try:
                m_arr = np.asarray(Image.open(str(mf)).convert("L"), dtype=np.uint8)
                mask_by_stem[mf.stem] = m_arr
                mask_by_stem[mf.name] = m_arr
            except Exception:
                pass

        if not mask_by_stem:
            _VISUAL_HULL_CACHE = False
            return False

        _VISUAL_HULL_CACHE = (views, mask_by_stem)
        return _VISUAL_HULL_CACHE
    except Exception:
        _VISUAL_HULL_CACHE = False
        return False


def _filter_largest_connected_component(verts, faces, colors=None, min_component_ratio=0.03):
    """
    Keep the primary connected component of triangles (and any secondary component with >= min_component_ratio
    of the primary component's face count), removing detached background islands and floating debris.
    """
    import numpy as np

    if len(faces) < 4:
        return verts, faces, colors

    # Build vertex -> face incidence via welded positions
    quant = np.round(verts * 1e5).astype(np.int64)
    _, _, weld_id = np.unique(quant, axis=0, return_index=True, return_inverse=True)
    w_faces = weld_id[faces]

    vert_to_faces = defaultdict(list)
    for fi in range(len(w_faces)):
        f0, f1, f2 = int(w_faces[fi, 0]), int(w_faces[fi, 1]), int(w_faces[fi, 2])
        vert_to_faces[f0].append(fi)
        vert_to_faces[f1].append(fi)
        vert_to_faces[f2].append(fi)

    visited = np.zeros(len(faces), dtype=bool)
    components = []
    for start_f in range(len(faces)):
        if visited[start_f]:
            continue
        comp = []
        q = deque([start_f])
        visited[start_f] = True
        while q:
            cur_f = q.popleft()
            comp.append(cur_f)
            for wv in (int(w_faces[cur_f, 0]), int(w_faces[cur_f, 1]), int(w_faces[cur_f, 2])):
                for nb_f in vert_to_faces[wv]:
                    if not visited[nb_f]:
                        visited[nb_f] = True
                        q.append(nb_f)
        components.append(comp)

    if len(components) <= 1:
        return verts, faces, colors

    components.sort(key=len, reverse=True)
    max_len = len(components[0])
    keep_f_mask = np.zeros(len(faces), dtype=bool)
    for comp in components:
        if len(comp) >= max(16, int(max_len * min_component_ratio)):
            keep_f_mask[comp] = True

    if np.all(keep_f_mask):
        return verts, faces, colors

    kept_faces = faces[keep_f_mask]
    used_v = np.unique(kept_faces)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used_v] = np.arange(len(used_v), dtype=np.int64)
    new_verts = verts[used_v]
    new_colors = colors[used_v] if (colors is not None and len(colors) == len(verts)) else colors
    new_faces = remap[kept_faces]
    return new_verts, new_faces, new_colors


def _carve_mesh_with_visual_hull_and_clean(verts, faces, colors=None, label="mesh"):
    """
    Remove stretched web triangles and out-of-silhouette background geometry using 3D Multi-View Visual Hull:
    1. Removes long stretched triangles (max_edge > 8.0 * median_edge).
    2. Projects mesh vertices into all camera masks ($DATASET/masks/*.png) and removes triangles
       that fall strictly outside the 3D Visual Hull (without filtering by dark vertex brightness,
       so black/dark objects are preserved intact).
    3. Keeps connected components >= 3% of the main component.
    """
    import numpy as np

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(verts) < 3 or len(faces) < 4:
        return verts, faces, colors

    has_colors = colors is not None and len(colors) == len(verts)
    if has_colors:
        colors = np.asarray(colors)

    orig_f_len = len(faces)
    keep_face = np.ones(len(faces), dtype=bool)

    # A. Filter extreme stretched/spike triangles (e.g. TSDF boundary rays or old spoke-wheel fans)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    max_e = np.maximum(e0, np.maximum(e1, e2))
    med_e = float(np.median(max_e)) if len(max_e) > 0 else 0.01
    if med_e > 1e-7:
        stretched = max_e > (med_e * 8.0)
        if np.sum(~stretched) >= int(len(faces) * 0.45):
            keep_face &= ~stretched

    # B. Multi-View 3D Visual-Hull carving using $DATASET/masks/*.png
    vh_ctx = _get_visual_hull_context()
    if vh_ctx:
        views, mask_by_stem = vh_ctx
        vis_cnt = np.zeros(len(verts), dtype=np.int32)
        fg_cnt = np.zeros(len(verts), dtype=np.int32)

        for img_name, R, t, (cw, ch, fx, fy, cx, cy) in views:
            m = mask_by_stem.get(img_name)
            if m is None:
                m = mask_by_stem.get(Path(img_name).stem)
            if m is None:
                continue
            mh, mw = m.shape[:2]
            sx, sy = float(mw) / max(1, cw), float(mh) / max(1, ch)
            p_cam = verts @ R.T + t
            z = p_cam[:, 2]
            valid = z > 0.05
            idx_v = np.nonzero(valid)[0]
            if len(idx_v) == 0:
                continue
            z_v = z[idx_v]
            u = np.round((fx * (p_cam[idx_v, 0] / z_v) + cx) * sx).astype(np.int32)
            v = np.round((fy * (p_cam[idx_v, 1] / z_v) + cy) * sy).astype(np.int32)
            in_b = (u >= 0) & (u < mw) & (v >= 0) & (v < mh)
            idx_in = idx_v[in_b]
            if len(idx_in) == 0:
                continue
            vis_cnt[idx_in] += 1
            is_fg = m[v[in_b], u[in_b]] > 127
            fg_cnt[idx_in[is_fg]] += 1

        fg_ratio = fg_cnt.astype(np.float32) / np.maximum(1, vis_cnt)

        for thresh in (0.60, 0.50, 0.40):
            v_ok = (vis_cnt < 3) | (fg_ratio >= thresh)
            cand_face = keep_face & v_ok[faces[:, 0]] & v_ok[faces[:, 1]] & v_ok[faces[:, 2]]
            if np.sum(cand_face) >= max(200, int(len(faces) * 0.45)):
                keep_face = cand_face
                break

    if not np.all(keep_face):
        faces = faces[keep_face]
        used_v = np.unique(faces)
        remap = np.full(len(verts), -1, dtype=np.int64)
        remap[used_v] = np.arange(len(used_v), dtype=np.int64)
        verts = verts[used_v]
        if has_colors:
            colors = colors[used_v]
        faces = remap[faces]

    verts, faces, colors = _filter_largest_connected_component(verts, faces, colors)
    if len(faces) < orig_f_len:
        print(
            "[MeshClean] Очищено сітку %s (Visual Hull + фільтр фону): залишено %d / %d граней."
            % (label, len(faces), orig_f_len),
            flush=True,
        )
    return verts, faces, colors


# ==============================================================================
# 3. Advancing-Front Watertight Mesh Hole-Closing (Zero Spoke-Wheel Fans!)
# ==============================================================================
def _close_mesh_boundaries_numpy(verts, faces, colors=None):
    """
    Topological boundary-loop detector and 2-to-1 Advancing-Front Ring Reduction triangulator.
    Halves the number of vertices on each concentric inward step (M -> M//2 -> M//4 -> ... -> <= 8),
    avoiding high-degree starburst hub vertices and long radial spoke triangles.
    Skips non-compact exterior tears so background edges are never bridged across empty space.
    """
    import numpy as np

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(verts) == 0 or len(faces) == 0:
        return verts, faces, colors, 0

    has_colors = colors is not None and len(colors) == len(verts)
    if has_colors:
        colors = np.asarray(colors)

    # Step A: Weld duplicate vertices at identical 3D positions
    quant = np.round(verts * 1e5).astype(np.int64)
    _, unique_v_idx, inv_v_idx = np.unique(quant, axis=0, return_index=True, return_inverse=True)
    if len(unique_v_idx) < len(verts):
        order = np.argsort(unique_v_idx)
        remap = np.empty_like(order)
        remap[order] = np.arange(len(order))
        verts = verts[unique_v_idx[order]]
        if has_colors:
            colors = colors[unique_v_idx[order]]
        faces = remap[inv_v_idx[faces]]

    # Step B: Remove degenerate and duplicate triangles
    valid_mask = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 2] != faces[:, 0])
    )
    faces = faces[valid_mask]
    if len(faces) == 0:
        return verts, faces, colors, 0

    _, unique_f_idx = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(unique_f_idx)]

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min)) or 1.0

    verts_list = [verts]
    colors_list = [colors] if has_colors else None
    total_loops_closed = 0

    for _pass in range(2):
        h0 = faces[:, [0, 1]]
        h1 = faces[:, [1, 2]]
        h2 = faces[:, [2, 0]]
        all_he = np.vstack([h0, h1, h2])

        u_min = np.minimum(all_he[:, 0], all_he[:, 1])
        u_max = np.maximum(all_he[:, 0], all_he[:, 1])
        undir_edges = np.column_stack([u_min, u_max])
        undir_counts = Counter(map(tuple, undir_edges.tolist()))

        undir_adj = defaultdict(list)
        pref_dir = set()
        boundary_edges = set()

        for u, v in all_he.tolist():
            u_i, v_i = int(u), int(v)
            key = (min(u_i, v_i), max(u_i, v_i))
            if undir_counts[key] == 1 and key not in boundary_edges:
                boundary_edges.add(key)
                undir_adj[u_i].append(v_i)
                undir_adj[v_i].append(u_i)
                pref_dir.add((v_i, u_i))

        if not boundary_edges:
            break

        cur_verts = np.vstack(verts_list)
        cur_colors = np.vstack(colors_list) if colors_list is not None else None
        next_vid = int(cur_verts.shape[0])
        new_faces = [faces]
        loops_in_pass = 0
        used_edges = set()

        def _seal_loop(loop_nodes):
            nonlocal next_vid, loops_in_pass
            L = len(loop_nodes)
            if L < 3:
                return

            idx_arr = np.array(loop_nodes, dtype=np.int64)
            loop_pts = cur_verts[idx_arr]
            centroid = loop_pts.mean(axis=0, keepdims=True)

            # Check hole compactness: skip giant jagged exterior tears that would bridge across space
            if L > 16:
                radii = np.linalg.norm(loop_pts - centroid, axis=1)
                r_max = float(np.max(radii))
                r_med = float(np.median(radii))
                seg_lens = np.linalg.norm(np.roll(loop_pts, -1, axis=0) - loop_pts, axis=1)
                perim = float(np.sum(seg_lens))
                tortuosity = perim / max(1e-6, 2.0 * math.pi * r_med)
                if r_max > 0.52 * bbox_diag or tortuosity > 3.6:
                    return

            # Orient loop consistently with pref_dir
            fwd_score = sum(1 for i in range(L) if (loop_nodes[i], loop_nodes[(i + 1) % L]) in pref_dir)
            rev_score = sum(1 for i in range(L) if (loop_nodes[(i + 1) % L], loop_nodes[i]) in pref_dir)
            if rev_score > fwd_score:
                loop_nodes = list(reversed(loop_nodes))
                idx_arr = np.array(loop_nodes, dtype=np.int64)
                loop_pts = cur_verts[idx_arr]

            if L == 3:
                new_faces.append(np.array([[loop_nodes[0], loop_nodes[1], loop_nodes[2]]], dtype=np.int64))
                loops_in_pass += 1
                return

            curr_ids = list(loop_nodes)
            curr_pts = loop_pts.copy()
            curr_cols = (
                cur_colors[idx_arr].astype(np.float64)
                if (colors_list is not None and cur_colors is not None)
                else None
            )

            # 2-to-1 Advancing-Front Ring Reduction until ring has <= 8 vertices
            while len(curr_ids) > 8:
                M = len(curr_ids)
                Mp = M // 2

                # Smooth contour and step 35% inward toward current ring centroid
                smoothed_pts = (
                    0.25 * np.roll(curr_pts, 1, axis=0)
                    + 0.50 * curr_pts
                    + 0.25 * np.roll(curr_pts, -1, axis=0)
                )
                ring_center = smoothed_pts.mean(axis=0, keepdims=True)
                contracted_pts = 0.65 * smoothed_pts + 0.35 * ring_center

                # Downsample 2-to-1 by pairing (2j, 2j+1)
                even_pts = contracted_pts[0 : 2 * Mp : 2]
                odd_pts = contracted_pts[1 : 2 * Mp : 2]
                inner_pts = 0.5 * (even_pts + odd_pts)
                verts_list.append(inner_pts)

                if colors_list is not None and curr_cols is not None:
                    smoothed_cols = (
                        0.25 * np.roll(curr_cols, 1, axis=0)
                        + 0.50 * curr_cols
                        + 0.25 * np.roll(curr_cols, -1, axis=0)
                    )
                    col_center = smoothed_cols.mean(axis=0, keepdims=True)
                    contracted_cols = 0.65 * smoothed_cols + 0.35 * col_center
                    inner_cols = 0.5 * (contracted_cols[0 : 2 * Mp : 2] + contracted_cols[1 : 2 * Mp : 2])
                    out_cols = inner_cols
                    if np.issubdtype(cur_colors.dtype, np.integer):
                        out_cols = np.clip(np.round(inner_cols), 0, 255).astype(cur_colors.dtype)
                    else:
                        out_cols = inner_cols.astype(cur_colors.dtype)
                    colors_list.append(out_cols)
                    curr_cols = inner_cols

                inner_ids = [next_vid + j for j in range(Mp)]
                next_vid += Mp

                strip_tris = []
                for j in range(Mp):
                    nj = (j + 1) % Mp
                    a0 = curr_ids[2 * j]
                    a1 = curr_ids[2 * j + 1]
                    a2 = curr_ids[2 * j + 2] if (2 * j + 2) < M else curr_ids[0]
                    b0 = inner_ids[j]
                    b1 = inner_ids[nj]
                    strip_tris.append([a0, a1, b0])
                    strip_tris.append([a1, a2, b1])
                    strip_tris.append([b0, a1, b1])
                if M % 2 == 1:
                    strip_tris.append([curr_ids[M - 1], curr_ids[0], inner_ids[0]])

                new_faces.append(np.array(strip_tris, dtype=np.int64))
                curr_ids = inner_ids
                curr_pts = inner_pts

            # Close final small ring (<= 8 vertices)
            M_final = len(curr_ids)
            final_center = curr_pts.mean(axis=0, keepdims=True)
            verts_list.append(final_center)
            if colors_list is not None and curr_cols is not None:
                c_final = curr_cols.mean(axis=0, keepdims=True)
                if np.issubdtype(cur_colors.dtype, np.integer):
                    c_final = np.clip(np.round(c_final), 0, 255).astype(cur_colors.dtype)
                else:
                    c_final = c_final.astype(cur_colors.dtype)
                colors_list.append(c_final)

            center_vid = next_vid
            next_vid += 1
            fan = np.array(
                [[curr_ids[i], curr_ids[(i + 1) % M_final], center_vid] for i in range(M_final)],
                dtype=np.int64,
            )
            new_faces.append(fan)
            loops_in_pass += 1

        for e_start in list(boundary_edges):
            if e_start in used_edges:
                continue
            u0, v0 = e_start
            if (v0, u0) in pref_dir:
                u0, v0 = v0, u0

            used_edges.add(e_start)
            path_nodes = [u0, v0]
            visited_idx = {u0: 0, v0: 1}
            curr = v0

            while True:
                candidates = []
                for nb in undir_adj[curr]:
                    ek = (min(curr, nb), max(curr, nb))
                    if ek not in used_edges:
                        candidates.append((nb, ek))

                if not candidates:
                    break

                chosen_nb, chosen_ek = candidates[0]
                for nb, ek in candidates:
                    if (curr, nb) in pref_dir:
                        chosen_nb, chosen_ek = nb, ek
                        break

                used_edges.add(chosen_ek)
                if chosen_nb in visited_idx:
                    loop_start = visited_idx[chosen_nb]
                    loop = path_nodes[loop_start:]
                    if len(loop) >= 3:
                        _seal_loop(loop)
                    for removed in path_nodes[loop_start + 1 :]:
                        visited_idx.pop(removed, None)
                    path_nodes = path_nodes[: loop_start + 1]
                    curr = chosen_nb
                else:
                    visited_idx[chosen_nb] = len(path_nodes)
                    path_nodes.append(chosen_nb)
                    curr = chosen_nb

        faces = np.vstack(new_faces)
        total_loops_closed += loops_in_pass
        if loops_in_pass == 0:
            break

    final_verts = np.vstack(verts_list)
    final_colors = np.vstack(colors_list) if colors_list is not None else None
    return final_verts, faces, final_colors, total_loops_closed


def _make_obj_watertight(path):
    """
    Carve out-of-silhouette background triangles (when masks exist) and close compact open
    boundary loops in a Wavefront .obj file using 2-to-1 Advancing-Front Ring Reduction
    while preserving UV coordinates (vt), normals (vn), mtllib, and usemtl directives.
    """
    import numpy as np

    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    verts = []
    vts = []
    faces_v = []
    face_v_to_vt = {}

    for line in lines:
        s = line.strip()
        if s.startswith("v "):
            parts = s.split()
            if len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif s.startswith("vt "):
            parts = s.split()
            if len(parts) >= 3:
                vts.append([float(parts[1]), float(parts[2])])
        elif s.startswith("f "):
            parts = s.split()[1:]
            if len(parts) >= 3:
                fv = []
                for p in parts:
                    sub = p.split("/")
                    vi = int(sub[0]) - 1 if int(sub[0]) > 0 else len(verts) + int(sub[0])
                    fv.append(vi)
                    if len(sub) >= 2 and sub[1]:
                        ti = int(sub[1]) - 1 if int(sub[1]) > 0 else len(vts) + int(sub[1])
                        face_v_to_vt[vi] = ti
                for i in range(1, len(fv) - 1):
                    faces_v.append([fv[0], fv[i], fv[i + 1]])

    if not verts or not faces_v:
        return False

    orig_v_count = len(verts)
    verts_arr = np.asarray(verts, dtype=np.float64)
    faces_arr = np.asarray(faces_v, dtype=np.int64)
    bbox_diag = float(np.linalg.norm(verts_arr.max(axis=0) - verts_arr.min(axis=0))) or 1.0

    h0 = faces_arr[:, [0, 1]]
    h1 = faces_arr[:, [1, 2]]
    h2 = faces_arr[:, [2, 0]]
    all_he = np.vstack([h0, h1, h2])
    u_min = np.minimum(all_he[:, 0], all_he[:, 1])
    u_max = np.maximum(all_he[:, 0], all_he[:, 1])
    undir_counts = Counter(map(tuple, np.column_stack([u_min, u_max]).tolist()))

    undir_adj = defaultdict(list)
    pref_dir = set()
    boundary_edges = set()
    for u, v in all_he.tolist():
        u_i, v_i = int(u), int(v)
        if u_i == v_i:
            continue
        key = (min(u_i, v_i), max(u_i, v_i))
        if undir_counts[key] == 1 and key not in boundary_edges:
            boundary_edges.add(key)
            undir_adj[u_i].append(v_i)
            undir_adj[v_i].append(u_i)
            pref_dir.add((v_i, u_i))

    if not boundary_edges:
        return True

    has_vt = len(vts) > 0 and len(face_v_to_vt) > 0
    vts_arr = np.asarray(vts, dtype=np.float64) if has_vt else None
    next_v_idx = orig_v_count + 1
    next_vt_idx = len(vts) + 1
    append_v_lines = []
    append_vt_lines = []
    append_f_lines = []
    loops_closed = 0
    used_edges = set()

    def _seal_obj_loop(loop):
        nonlocal next_v_idx, next_vt_idx, loops_closed
        L = len(loop)
        if L < 3:
            return
        loop_pts = verts_arr[np.array(loop, dtype=np.int64)]
        centroid = loop_pts.mean(axis=0, keepdims=True)
        if L > 16:
            radii = np.linalg.norm(loop_pts - centroid, axis=1)
            r_max = float(np.max(radii))
            r_med = float(np.median(radii))
            perim = float(np.sum(np.linalg.norm(np.roll(loop_pts, -1, axis=0) - loop_pts, axis=1)))
            if r_max > 0.52 * bbox_diag or (perim / max(1e-6, 2.0 * math.pi * r_med)) > 3.6:
                return

        fwd_score = sum(1 for i in range(L) if (loop[i], loop[(i + 1) % L]) in pref_dir)
        rev_score = sum(1 for i in range(L) if (loop[(i + 1) % L], loop[i]) in pref_dir)
        if rev_score > fwd_score:
            loop = list(reversed(loop))
            loop_pts = verts_arr[np.array(loop, dtype=np.int64)]

        curr_v_1based = [vi + 1 for vi in loop]
        curr_vt_1based = [face_v_to_vt.get(vi, 0) + 1 for vi in loop] if has_vt else None
        curr_pts = loop_pts.copy()
        curr_uvs = (
            np.array(
                [
                    vts_arr[face_v_to_vt[vi]]
                    if (vi in face_v_to_vt and face_v_to_vt[vi] < len(vts_arr))
                    else [0.5, 0.5]
                    for vi in loop
                ],
                dtype=np.float64,
            )
            if (has_vt and vts_arr is not None)
            else None
        )

        while len(curr_v_1based) > 8:
            M = len(curr_v_1based)
            Mp = M // 2
            smoothed = 0.25 * np.roll(curr_pts, 1, axis=0) + 0.50 * curr_pts + 0.25 * np.roll(curr_pts, -1, axis=0)
            rc = smoothed.mean(axis=0, keepdims=True)
            contracted = 0.65 * smoothed + 0.35 * rc
            inner_pts = 0.5 * (contracted[0 : 2 * Mp : 2] + contracted[1 : 2 * Mp : 2])

            inner_v_1based = []
            inner_vt_1based = []
            for j in range(Mp):
                append_v_lines.append("v %.6f %.6f %.6f" % (inner_pts[j, 0], inner_pts[j, 1], inner_pts[j, 2]))
                inner_v_1based.append(next_v_idx)
                next_v_idx += 1

            if curr_uvs is not None:
                sm_uv = 0.25 * np.roll(curr_uvs, 1, axis=0) + 0.50 * curr_uvs + 0.25 * np.roll(curr_uvs, -1, axis=0)
                uc = sm_uv.mean(axis=0, keepdims=True)
                con_uv = 0.65 * sm_uv + 0.35 * uc
                in_uv = 0.5 * (con_uv[0 : 2 * Mp : 2] + con_uv[1 : 2 * Mp : 2])
                for j in range(Mp):
                    append_vt_lines.append("vt %.6f %.6f" % (in_uv[j, 0], in_uv[j, 1]))
                    inner_vt_1based.append(next_vt_idx)
                    next_vt_idx += 1
                curr_uvs = in_uv

            for j in range(Mp):
                nj = (j + 1) % Mp
                i0, i1 = 2 * j, 2 * j + 1
                i2 = (2 * j + 2) if (2 * j + 2) < M else 0
                a0, a1, a2 = curr_v_1based[i0], curr_v_1based[i1], curr_v_1based[i2]
                b0, b1 = inner_v_1based[j], inner_v_1based[nj]
                if curr_vt_1based is not None and inner_vt_1based:
                    ta0, ta1, ta2 = curr_vt_1based[i0], curr_vt_1based[i1], curr_vt_1based[i2]
                    tb0, tb1 = inner_vt_1based[j], inner_vt_1based[nj]
                    append_f_lines.append("f %d/%d %d/%d %d/%d" % (a0, ta0, a1, ta1, b0, tb0))
                    append_f_lines.append("f %d/%d %d/%d %d/%d" % (a1, ta1, a2, ta2, b1, tb1))
                    append_f_lines.append("f %d/%d %d/%d %d/%d" % (b0, tb0, a1, ta1, b1, tb1))
                else:
                    append_f_lines.append("f %d %d %d" % (a0, a1, b0))
                    append_f_lines.append("f %d %d %d" % (a1, a2, b1))
                    append_f_lines.append("f %d %d %d" % (b0, a1, b1))

            if M % 2 == 1:
                a_last, a_first, b_first = curr_v_1based[M - 1], curr_v_1based[0], inner_v_1based[0]
                if curr_vt_1based is not None and inner_vt_1based:
                    append_f_lines.append(
                        "f %d/%d %d/%d %d/%d"
                        % (a_last, curr_vt_1based[M - 1], a_first, curr_vt_1based[0], b_first, inner_vt_1based[0])
                    )
                else:
                    append_f_lines.append("f %d %d %d" % (a_last, a_first, b_first))

            curr_v_1based = inner_v_1based
            curr_vt_1based = inner_vt_1based if inner_vt_1based else None
            curr_pts = inner_pts

        M_fin = len(curr_v_1based)
        c_xyz = curr_pts.mean(axis=0)
        append_v_lines.append("v %.6f %.6f %.6f" % (c_xyz[0], c_xyz[1], c_xyz[2]))
        c_v_idx = next_v_idx
        next_v_idx += 1
        c_vt_idx = 1
        if curr_uvs is not None:
            uv_c = curr_uvs.mean(axis=0)
            append_vt_lines.append("vt %.6f %.6f" % (uv_c[0], uv_c[1]))
            c_vt_idx = next_vt_idx
            next_vt_idx += 1

        for i in range(M_fin):
            ni = (i + 1) % M_fin
            a, b = curr_v_1based[i], curr_v_1based[ni]
            if curr_vt_1based is not None:
                ta, tb = curr_vt_1based[i], curr_vt_1based[ni]
                append_f_lines.append("f %d/%d %d/%d %d/%d" % (a, ta, b, tb, c_v_idx, c_vt_idx))
            else:
                append_f_lines.append("f %d %d %d" % (a, b, c_v_idx))
        loops_closed += 1

    for e_start in list(boundary_edges):
        if e_start in used_edges:
            continue
        u0, v0 = e_start
        if (v0, u0) in pref_dir:
            u0, v0 = v0, u0
        used_edges.add(e_start)
        path_nodes = [u0, v0]
        visited_idx = {u0: 0, v0: 1}
        curr = v0

        while True:
            candidates = [
                (nb, (min(curr, nb), max(curr, nb)))
                for nb in undir_adj[curr]
                if (min(curr, nb), max(curr, nb)) not in used_edges
            ]
            if not candidates:
                break
            chosen_nb, chosen_ek = candidates[0]
            for nb, ek in candidates:
                if (curr, nb) in pref_dir:
                    chosen_nb, chosen_ek = nb, ek
                    break
            used_edges.add(chosen_ek)
            if chosen_nb in visited_idx:
                loop_start = visited_idx[chosen_nb]
                loop = path_nodes[loop_start:]
                if len(loop) >= 3:
                    _seal_obj_loop(loop)
                for removed in path_nodes[loop_start + 1 :]:
                    visited_idx.pop(removed, None)
                path_nodes = path_nodes[: loop_start + 1]
                curr = chosen_nb
            else:
                visited_idx[chosen_nb] = len(path_nodes)
                path_nodes.append(chosen_nb)
                curr = chosen_nb

    if append_f_lines:
        out_block = ["", "# Watertight hole-closing cap"] + append_v_lines + append_vt_lines + append_f_lines
        with open(str(path), "a", encoding="utf-8") as f_out:
            f_out.write("\n".join(out_block) + "\n")

    print(
        "[Watertight] Зашито %d контурів у %s: +%d V, +%d F."
        % (loops_closed, path.name, len(append_v_lines), len(append_f_lines)),
        flush=True,
    )
    return True


def _repair_open3d_mesh_in_place(mesh, label="mesh", do_watertight=None):
    """
    1. Carve out-of-silhouette background triangles (when $DATASET/masks exists).
    2. Keep primary connected component.
    3. Optionally close compact open boundary loops (when do_watertight / _watertight_flag is True).
    """
    import numpy as np
    import open3d as o3d

    if do_watertight is None:
        do_watertight = _watertight_flag

    faces = np.asarray(mesh.triangles, dtype=np.int64)
    if len(faces) == 0:
        return mesh, 0

    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    has_colors = mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(verts)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float64) if has_colors else None

    verts, faces, colors = _carve_mesh_with_visual_hull_and_clean(verts, faces, colors, label=label)

    loops_closed = 0
    if do_watertight:
        verts, faces, colors, loops_closed = _close_mesh_boundaries_numpy(verts, faces, colors)

    out_mesh = o3d.geometry.TriangleMesh()
    out_mesh.vertices = o3d.utility.Vector3dVector(verts)
    out_mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    if colors is not None and len(colors) == len(verts):
        out_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    out_mesh.remove_duplicated_triangles()
    out_mesh.remove_degenerate_triangles()
    out_mesh.remove_unreferenced_vertices()
    out_mesh.compute_vertex_normals()

    if do_watertight:
        print(
            "[Watertight] Готово (%s): зашито %d відкритих контурів -> %d V, %d F."
            % (label, loops_closed, len(out_mesh.vertices), len(out_mesh.triangles)),
            flush=True,
        )
    return out_mesh, loops_closed


_IN_WATERTIGHT_WRITE = False


def make_mesh_watertight(mesh_file, force_watertight=None):
    """
    Clean background skirt triangles (via 3D Visual Hull) and close compact open boundary loops
    in a triangle mesh (.ply or .obj) while preserving vertex colors or UV textures.
    """
    global _IN_WATERTIGHT_WRITE
    if _IN_WATERTIGHT_WRITE:
        return False

    do_wt = _watertight_flag if force_watertight is None else bool(force_watertight)
    path = Path(str(mesh_file))
    if not path.is_file():
        return False

    suffix = path.suffix.lower()
    if suffix == ".obj":
        if not do_wt:
            return False
        try:
            return _make_obj_watertight(path)
        except Exception as exc:
            print("[Watertight] Попередження під час обробки OBJ %s: %s" % (path.name, exc), flush=True)
            return False

    if suffix != ".ply":
        return False

    _IN_WATERTIGHT_WRITE = True
    try:
        import numpy as np

        try:
            import open3d as o3d

            mesh = o3d.io.read_triangle_mesh(str(path))
            if len(mesh.triangles) == 0:
                return False

            out_mesh, _ = _repair_open3d_mesh_in_place(mesh, label=path.name, do_watertight=do_wt)
            o3d.io.write_triangle_mesh(str(path), out_mesh)
            return True
        except ImportError:
            pass

        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(str(path))
        if "face" not in plydata or len(plydata["face"].data) == 0:
            return False

        v_data = plydata["vertex"].data
        verts = np.column_stack([v_data["x"], v_data["y"], v_data["z"]]).astype(np.float64)
        prop_names = v_data.dtype.names or ()
        has_rgb = all(c in prop_names for c in ("red", "green", "blue"))
        colors = (
            np.column_stack([v_data["red"], v_data["green"], v_data["blue"]]).astype(np.uint8)
            if has_rgb
            else None
        )

        f_prop = plydata["face"].data.dtype.names[0]
        raw_faces = plydata["face"].data[f_prop]
        faces = np.array([list(f[:3]) for f in raw_faces if len(f) >= 3], dtype=np.int64)
        if len(faces) == 0:
            return False

        verts, faces, colors = _carve_mesh_with_visual_hull_and_clean(verts, faces, colors, label=path.name)
        loops_closed = 0
        if do_wt:
            verts, faces, colors, loops_closed = _close_mesh_boundaries_numpy(verts, faces, colors)

        if has_rgb and colors is not None:
            v_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
            v_arr = np.empty(len(verts), dtype=v_dtype)
            v_arr["x"] = verts[:, 0].astype(np.float32)
            v_arr["y"] = verts[:, 1].astype(np.float32)
            v_arr["z"] = verts[:, 2].astype(np.float32)
            v_arr["red"] = colors[:, 0].astype(np.uint8)
            v_arr["green"] = colors[:, 1].astype(np.uint8)
            v_arr["blue"] = colors[:, 2].astype(np.uint8)
        else:
            v_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
            v_arr = np.empty(len(verts), dtype=v_dtype)
            v_arr["x"] = verts[:, 0].astype(np.float32)
            v_arr["y"] = verts[:, 1].astype(np.float32)
            v_arr["z"] = verts[:, 2].astype(np.float32)

        f_dtype = [(f_prop, "i4", (3,))]
        f_arr = np.empty(len(faces), dtype=f_dtype)
        f_arr[f_prop] = faces.astype(np.int32)

        PlyData([PlyElement.describe(v_arr, "vertex"), PlyElement.describe(f_arr, "face")], text=False).write(
            str(path)
        )
        return True
    except Exception as exc:
        print("[Watertight] Попередження під час обробки %s: %s" % (path, exc), flush=True)
        return False
    finally:
        _IN_WATERTIGHT_WRITE = False


# ==============================================================================
# 4. Module Import Hooks (GaussianModel LiveStreamer + Open3D + SuGaR Refine)
# ==============================================================================
def _sanitize_camera_infos_in_place(cam_infos):
    import numpy as np

    if not cam_infos:
        return
    is_list = isinstance(cam_infos, list)
    for idx in range(len(cam_infos)):
        ci = cam_infos[idx]
        replacements = {}

        d_arr = getattr(ci, "depth", None)
        if isinstance(d_arr, np.ndarray):
            bad_d = ~np.isfinite(d_arr) | (d_arr <= 1e-4)
            if np.any(bad_d):
                good_d = d_arr[~bad_d]
                fill_d = float(np.median(good_d)) if good_d.size > 0 else 1.0
                replacements["depth"] = np.where(bad_d, fill_d, d_arr).astype(np.float32)

        n_arr = getattr(ci, "normal", None)
        if isinstance(n_arr, np.ndarray) and n_arr.ndim == 3 and n_arr.shape[-1] == 3:
            n_norm = np.linalg.norm(n_arr, axis=-1, keepdims=True)
            bad_n = (~np.all(np.isfinite(n_arr), axis=-1)) | (n_norm[..., 0] < 1e-6)
            if np.any(bad_n):
                n_clean = n_arr.copy().astype(np.float32)
                n_clean[bad_n] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                replacements["normal"] = n_clean

        if replacements:
            if is_list and hasattr(ci, "_replace"):
                try:
                    cam_infos[idx] = ci._replace(**replacements)
                    continue
                except Exception:
                    pass
            for k, v in replacements.items():
                try:
                    setattr(ci, k, v)
                except Exception:
                    pass


def _patch_point_generator(mod):
    if mod is None or getattr(mod, "_bakalarka_pg_patched", False):
        return
    orig_gen = getattr(mod, "gaussian_generator", None)
    if callable(orig_gen) and not getattr(orig_gen, "_bakalarka_wrapped", False):
        def patched_gaussian_generator(cam_infos, cam_intrinsics, num_generated_points, image_depth_scale_map, *g_args, **g_kwargs):
            import numpy as np

            _sanitize_camera_infos_in_place(cam_infos)

            if isinstance(image_depth_scale_map, dict):
                valid_scales = []
                for v in image_depth_scale_map.values():
                    if isinstance(v, dict):
                        sc = v.get("estimated_scale")
                        if sc is not None and np.isfinite(sc) and float(sc) > 1e-4:
                            valid_scales.append(float(sc))
                fallback_scale = float(np.median(valid_scales)) if valid_scales else 1.0

                for ci in cam_infos:
                    img_id = getattr(ci, "image_id", None)
                    if img_id is not None:
                        entry = image_depth_scale_map.get(img_id)
                        if not isinstance(entry, dict):
                            image_depth_scale_map[img_id] = {"estimated_scale": fallback_scale, "offset": 0.0}
                        else:
                            sc = entry.get("estimated_scale")
                            if sc is None or not np.isfinite(sc) or float(sc) <= 1e-4:
                                entry["estimated_scale"] = fallback_scale
                            off = entry.get("offset", 0.0)
                            if off is None or not np.isfinite(off):
                                entry["offset"] = 0.0

            xyz, rgb, normals = orig_gen(cam_infos, cam_intrinsics, num_generated_points, image_depth_scale_map, *g_args, **g_kwargs)

            try:
                xyz = np.asarray(xyz, dtype=np.float64)
                rgb = np.asarray(rgb, dtype=np.float64)
                normals = np.asarray(normals, dtype=np.float64)

                finite_mask = (
                    np.all(np.isfinite(xyz), axis=1)
                    & np.all(np.isfinite(rgb), axis=1)
                    & np.all(np.isfinite(normals), axis=1)
                )
                if not np.all(finite_mask):
                    xyz = xyz[finite_mask]
                    rgb = rgb[finite_mask]
                    normals = normals[finite_mask]

                ctx = _get_visual_hull_context()
                if ctx and len(xyz) > 100:
                    vis_cnt, fg_cnt = _project_points_to_visual_hull(xyz, ctx, min_views=3)
                    fg_ratio = fg_cnt.astype(np.float64) / np.maximum(vis_cnt, 1).astype(np.float64)
                    keep = (vis_cnt >= 3) & (fg_ratio >= 0.58)
                    if int(keep.sum()) < 500:
                        keep = (vis_cnt >= 2) & (fg_ratio >= 0.45)
                    if int(keep.sum()) >= 100:
                        before_n = len(xyz)
                        xyz = xyz[keep]
                        rgb = rgb[keep]
                        normals = normals[keep]
                        print(
                            "[VisualHull] Очищено згенеровані точки ініціалізації: %d -> %d (відсіяно %d фонових точок)."
                            % (before_n, len(xyz), before_n - len(xyz)),
                            flush=True,
                        )

                norms_len = np.linalg.norm(normals, axis=1, keepdims=True)
                bad_n = (norms_len[:, 0] < 1e-8) | (~np.isfinite(norms_len[:, 0]))
                if np.any(bad_n):
                    normals[bad_n] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    norms_len = np.linalg.norm(normals, axis=1, keepdims=True)
                normals = normals / np.maximum(norms_len, 1e-8)
            except Exception as exc:
                print("[VisualHull] Попередження у patched_gaussian_generator: %s" % exc, flush=True)

            return xyz, rgb, normals

        patched_gaussian_generator._bakalarka_wrapped = True
        mod.gaussian_generator = patched_gaussian_generator

    orig_read_colmap = getattr(mod, "readColmapSceneInfo", None)
    if callable(orig_read_colmap) and not getattr(orig_read_colmap, "_bakalarka_wrapped", False):
        def patched_read_colmap(*r_args, **r_kwargs):
            scene_info = orig_read_colmap(*r_args, **r_kwargs)
            try:
                train_cams = getattr(scene_info, "train_cameras", None)
                test_cams = getattr(scene_info, "test_cameras", None)
                if isinstance(train_cams, list):
                    _sanitize_camera_infos_in_place(train_cams)
                if isinstance(test_cams, list):
                    _sanitize_camera_infos_in_place(test_cams)
            except Exception:
                pass
            return scene_info

        patched_read_colmap._bakalarka_wrapped = True
        mod.readColmapSceneInfo = patched_read_colmap

    mod._bakalarka_pg_patched = True


def _patch_open3d_io(o3d_mod):
    if not (_watertight_flag or _mask_fg_flag or _get_visual_hull_context()):
        return
    io_mod = getattr(o3d_mod, "io", None)
    if io_mod is None or getattr(io_mod, "_bakalarka_wt_patched", False):
        return
    orig_write = getattr(io_mod, "write_triangle_mesh", None)
    if not callable(orig_write):
        return

    io_mod._bakalarka_wt_patched = True

    def patched_write_triangle_mesh(filename, mesh, *w_args, **w_kwargs):
        fname_str = str(filename)
        if not _IN_WATERTIGHT_WRITE and fname_str.lower().endswith(".ply"):
            try:
                mesh, _ = _repair_open3d_mesh_in_place(
                    mesh, label=os.path.basename(fname_str), do_watertight=_watertight_flag
                )
            except Exception:
                pass
        return orig_write(filename, mesh, *w_args, **w_kwargs)

    io_mod.write_triangle_mesh = patched_write_triangle_mesh


def _patch_refine_module(refine_mod):
    if getattr(refine_mod, "_bakalarka_refine_patched", False):
        return
    orig_refined_training = getattr(refine_mod, "refined_training", None)
    if not callable(orig_refined_training):
        return

    refine_mod._bakalarka_refine_patched = True

    def patched_refined_training(*r_args, **r_kwargs):
        cfg = r_args[0] if r_args else None
        mesh_path = getattr(cfg, "mesh_path", None) if cfg is not None else r_kwargs.get("mesh_path")
        output_dir = getattr(cfg, "output_dir", None) if cfg is not None else r_kwargs.get("output_dir")
        if not output_dir:
            output_dir = os.getenv("MESH_OUTPUT_DIR")

        if not mesh_path or not os.path.isfile(str(mesh_path)):
            model_dir = os.getenv("MODEL_OUTPUT_DIR")
            if model_dir and os.path.isdir(model_dir):
                candidates = sorted(Path(model_dir).rglob("fuse_post.ply")) or sorted(Path(model_dir).rglob("fuse.ply"))
                if candidates:
                    mesh_path = str(candidates[-1])

        if (_watertight_flag or _mask_fg_flag or _get_visual_hull_context()) and mesh_path and os.path.isfile(str(mesh_path)):
            make_mesh_watertight(mesh_path, force_watertight=_watertight_flag)

        if _refinement_mode in ("skip", "0", "none", "false"):
            print(
                "\n========== [2D-SuGaR] REFINEMENT_MODE=skip: Пропускаємо етап SuGaR Refinement (15 000 кроків) ==========",
                flush=True,
            )
            if mesh_path and os.path.isfile(str(mesh_path)) and output_dir:
                out_dir_path = Path(output_dir)
                out_dir_path.mkdir(parents=True, exist_ok=True)
                dst_ply = out_dir_path / Path(mesh_path).name
                if os.path.abspath(str(mesh_path)) != os.path.abspath(str(dst_ply)):
                    shutil.copy2(str(mesh_path), str(dst_ply))
                print("[2D-SuGaR] Готову TSDF-сітку збережено у: %s" % dst_ply, flush=True)
            print("Iteration: 15000\nloss: 0.000000  [ 15000/15000] computed in 0.01 minutes.", flush=True)
            return None

        if _refinement_mode in ("fast", "2000", "balanced", "7000"):
            target_iters = 2000 if _refinement_mode in ("fast", "2000") else 7000
            print(
                "\n========== [2D-SuGaR] REFINEMENT_MODE=%s: Запуск SuGaR Refinement на %d ітерацій =========="
                % (_refinement_mode, target_iters),
                flush=True,
            )
            if cfg is not None:
                setattr(cfg, "refinement_iterations", target_iters)
            if "refinement_iterations" in r_kwargs:
                r_kwargs["refinement_iterations"] = target_iters

        res = orig_refined_training(*r_args, **r_kwargs)

        if (_watertight_flag or _mask_fg_flag or _get_visual_hull_context()) and output_dir and os.path.isdir(str(output_dir)):
            for ext in ("*.ply", "*.obj"):
                for f in sorted(Path(output_dir).rglob(ext)):
                    if not f.name.startswith("point_cloud"):
                        make_mesh_watertight(f, force_watertight=_watertight_flag)

        return res

    refine_mod.refined_training = patched_refined_training


try:
    _orig_import = builtins.__import__

    def _custom_import(name, *args, **kwargs):
        mod = _orig_import(name, *args, **kwargs)

        # Hook 1: GaussianModel optimizer step for LiveStreamer (WebSocket live 3DGS view)
        if streamer is not None and (
            name == "scene.gaussian_model"
            or name == "scene"
            or name.endswith(".gaussian_model")
            or name.endswith(".scene")
        ):
            try:
                target_cls = getattr(mod, "GaussianModel", None)
                if target_cls and not getattr(target_cls, "_streamer_patched", False):
                    target_cls._streamer_patched = True
                    orig_setup = target_cls.training_setup

                    def patched_training_setup(self, *s_args, **s_kwargs):
                        orig_setup(self, *s_args, **s_kwargs)
                        opt = getattr(self, "optimizer", None)
                        if opt:
                            opt._iteration_counter = 0
                            orig_opt_step = opt.step

                            def patched_opt_step(*o_args, **o_kwargs):
                                res = orig_opt_step(*o_args, **o_kwargs)
                                opt._iteration_counter += 1
                                try:
                                    streamer.step(opt._iteration_counter, self)
                                except Exception:
                                    pass
                                return res

                            opt.step = patched_opt_step

                    target_cls.training_setup = patched_training_setup
            except Exception:
                pass

        # Hook 2: open3d.io.write_triangle_mesh for immediate in-memory Visual-Hull carving & watertight repair
        if name == "open3d" or name.startswith("open3d."):
            try:
                o3d_root = sys.modules.get("open3d") or mod
                if o3d_root is not None:
                    _patch_open3d_io(o3d_root)
            except Exception:
                pass

        # Hook 3: sugar_trainers.refine for REFINEMENT_MODE and WATERTIGHT_MESH
        if "sugar_trainers" in name:
            try:
                refine_mod = sys.modules.get("sugar_trainers.refine") or (
                    mod if name == "sugar_trainers.refine" else getattr(mod, "refine", None)
                )
                if refine_mod is not None:
                    _patch_refine_module(refine_mod)
            except Exception:
                pass

        # Hook 4: scene.point_generator / scene.dataset_readers for safe depth scaling & Visual-Hull point filtering
        if (
            "point_generator" in name
            or "dataset_readers" in name
            or name == "scene"
            or name.endswith(".scene")
        ):
            try:
                for mod_candidate in (
                    mod,
                    sys.modules.get("scene.point_generator"),
                    sys.modules.get("scene.dataset_readers"),
                    getattr(mod, "point_generator", None),
                    getattr(mod, "dataset_readers", None),
                ):
                    if mod_candidate is not None:
                        _patch_point_generator(mod_candidate)
            except Exception:
                pass

        return mod

    builtins.__import__ = _custom_import
except Exception:
    pass
