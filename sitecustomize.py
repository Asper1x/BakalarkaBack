"""
Automatic hook for 3D Gaussian Splatting / 2D-SuGaR training:
1. Non-blocking LiveStreamer telemetry & point-cloud streaming (when LIVE_STREAM_PACKET is set)
2. Configurable SuGaR Mesh Refinement mode (REFINEMENT_MODE = skip | fast | balanced | full)
3. Automatic Watertight Mesh hole-filling & manifold repair (when WATERTIGHT_MESH=1)
Loaded automatically via PYTHONPATH by Python site module.
Compatible with Python 3.7+ (no PEP 604 union types).
"""
import builtins
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
# 2. Resolve REFINEMENT_MODE and WATERTIGHT_MESH (including composite SCENE_TYPE)
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
    Lazily load COLMAP camera poses and 2D foreground masks (masks/*.png)
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


def _project_points_to_visual_hull(points_xyz, ctx):
    import numpy as np

    views, mask_by_stem = ctx
    n_pts = len(points_xyz)
    vis_cnt = np.zeros(n_pts, dtype=np.int32)
    fg_cnt = np.zeros(n_pts, dtype=np.int32)
    if n_pts == 0 or not views:
        return vis_cnt, fg_cnt

    for img_name, R, t, (cw, ch, fx, fy, cx, cy) in views:
        m = mask_by_stem.get(img_name)
        if m is None:
            m = mask_by_stem.get(Path(img_name).stem)
        if m is None:
            continue
        mh, mw = m.shape[:2]
        sx, sy = float(mw) / max(1, cw), float(mh) / max(1, ch)
        p_cam = points_xyz @ R.T + t
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

    return vis_cnt, fg_cnt


# ==============================================================================
# 3. Watertight Mesh Hole-Closing (Open3D + Advancing-Front 2-to-1 Ring Reduction)
# ==============================================================================
def _close_mesh_boundaries_numpy(verts, faces, colors=None):
    """
    Topological boundary-loop detector and 2-to-1 advancing-front membrane triangulator.
    Detects open boundary edges via undirected edge triangle incidence == 1 across
    multi-scale spatial welding passes, prunes dead-end boundary spurs, traces both
    closed boundary cycles and open boundary chains, and seals both small holes and
    large base openings using 2-to-1 ring reduction (M -> M // 2) so no vertex ever
    exceeds degree 8 (preventing starburst/spoke-wheel artifacts).
    Returns (repaired_verts, repaired_faces, repaired_colors, holes_closed_count).
    """
    import math
    import numpy as np
    from collections import Counter, defaultdict

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(verts) == 0 or len(faces) == 0:
        return verts, faces, colors, 0

    has_colors = colors is not None and len(colors) == len(verts)
    if has_colors:
        colors = np.asarray(colors)

    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
    if bbox_diag < 1e-6:
        bbox_diag = 1.0

    # Step A: Exact-tolerance vertex welding (TSDF block seams)
    quant = np.round((verts - bbox_min) * (25000.0 / bbox_diag)).astype(np.int64)
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

    verts_list = [verts]
    colors_list = [colors] if has_colors else None
    total_loops_closed = 0

    quant_factors = (
        25000.0 / bbox_diag,
        3000.0 / bbox_diag,
        600.0 / bbox_diag,
        180.0 / bbox_diag,
    )

    for q_factor in quant_factors:
        cur_verts = np.vstack(verts_list)
        cur_colors = np.vstack(colors_list) if colors_list is not None else None

        # Build spatial weld mapping for boundary detection at current scale
        q_coords = np.round((cur_verts - bbox_min) * q_factor).astype(np.int64)
        _, u_idx, weld_of = np.unique(q_coords, axis=0, return_index=True, return_inverse=True)
        rep_vid = u_idx[weld_of]

        w_faces = rep_vid[faces]
        valid_wf = (
            (w_faces[:, 0] != w_faces[:, 1])
            & (w_faces[:, 1] != w_faces[:, 2])
            & (w_faces[:, 2] != w_faces[:, 0])
        )
        w_faces_clean = w_faces[valid_wf]
        if len(w_faces_clean) == 0:
            break

        h0 = w_faces_clean[:, [0, 1]]
        h1 = w_faces_clean[:, [1, 2]]
        h2 = w_faces_clean[:, [2, 0]]
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
            if u_i == v_i:
                continue
            key = (min(u_i, v_i), max(u_i, v_i))
            if undir_counts[key] == 1 and key not in boundary_edges:
                boundary_edges.add(key)
                undir_adj[u_i].append(v_i)
                undir_adj[v_i].append(u_i)
                # Existing triangle has u_i -> v_i, so hole cap needs v_i -> u_i
                pref_dir.add((v_i, u_i))

        if not boundary_edges:
            break

        # Prune dead-end degree-1 spurs only where they attach to branch vertices (degree >= 3)
        used_edges = set()
        for _prune_it in range(8):
            pruned_any = False
            for u, nbs in list(undir_adj.items()):
                alive = [
                    nb for nb in nbs
                    if (min(u, nb), max(u, nb)) not in used_edges
                ]
                if len(alive) == 1:
                    nb = alive[0]
                    nb_alive = [
                        w for w in undir_adj[nb]
                        if (min(nb, w), max(nb, w)) not in used_edges
                    ]
                    if len(nb_alive) >= 3:
                        used_edges.add((min(u, nb), max(u, nb)))
                        pruned_any = True
            if not pruned_any:
                break

        next_vid = int(cur_verts.shape[0])
        new_faces = [faces]
        loops_in_pass = 0

        def _seal_loop(raw_loop_nodes):
            nonlocal next_vid, loops_in_pass
            # Deduplicate consecutive repeated nodes and collapse immediate u -> v -> u hairpins
            loop_nodes = []
            for nid in raw_loop_nodes:
                nid_i = int(nid)
                if loop_nodes and loop_nodes[-1] == nid_i:
                    continue
                if len(loop_nodes) >= 2 and loop_nodes[-2] == nid_i:
                    loop_nodes.pop()
                    continue
                loop_nodes.append(nid_i)
            while len(loop_nodes) > 2 and loop_nodes[0] == loop_nodes[-1]:
                loop_nodes.pop()

            L0 = len(loop_nodes)
            if L0 < 3:
                return

            # Orient loop consistently with pref_dir
            fwd_score = sum(
                1 for i in range(L0)
                if (loop_nodes[i], loop_nodes[(i + 1) % L0]) in pref_dir
            )
            rev_score = sum(
                1 for i in range(L0)
                if (loop_nodes[(i + 1) % L0], loop_nodes[i]) in pref_dir
            )
            if rev_score > fwd_score:
                loop_nodes = list(reversed(loop_nodes))

            if L0 == 3:
                new_faces.append(
                    np.array([[loop_nodes[0], loop_nodes[1], loop_nodes[2]]], dtype=np.int64)
                )
                loops_in_pass += 1
                return

            if L0 == 4:
                p0, p1, p2, p3 = [cur_verts[idx] for idx in loop_nodes]
                if np.linalg.norm(p0 - p2) <= np.linalg.norm(p1 - p3):
                    new_faces.append(
                        np.array(
                            [
                                [loop_nodes[0], loop_nodes[1], loop_nodes[2]],
                                [loop_nodes[0], loop_nodes[2], loop_nodes[3]],
                            ],
                            dtype=np.int64,
                        )
                    )
                else:
                    new_faces.append(
                        np.array(
                            [
                                [loop_nodes[0], loop_nodes[1], loop_nodes[3]],
                                [loop_nodes[1], loop_nodes[2], loop_nodes[3]],
                            ],
                            dtype=np.int64,
                        )
                    )
                loops_in_pass += 1
                return

            # Subdivide any long bridged gap (> 2.2x median loop edge) so all sides of the hole
            # have uniform vertex density before 2-to-1 ring reduction
            pts0 = cur_verts[np.array(loop_nodes, dtype=np.int64)]
            seg_d = np.linalg.norm(np.roll(pts0, -1, axis=0) - pts0, axis=1)
            valid_d = seg_d[seg_d > 1e-6]
            med_edge = float(np.median(valid_d)) if len(valid_d) > 0 else bbox_diag * 0.015

            # Shortcut narrow outward spike excursions along the loop so the cap never stretches webs out to spikes
            if len(loop_nodes) >= 16:
                for _ex_pass in range(2):
                    n_b = len(loop_nodes)
                    if n_b < 16:
                        break
                    max_span = min(64, n_b // 4)
                    pruned_loop = []
                    idx_l = 0
                    while idx_l < n_b:
                        p_a = cur_verts[loop_nodes[idx_l]]
                        jump_to = -1
                        arc_acc = 0.0
                        prev_p = p_a
                        lim = min(max_span, n_b - 1 - idx_l)
                        for s in range(1, lim + 1):
                            p_b = cur_verts[loop_nodes[idx_l + s]]
                            arc_acc += float(np.linalg.norm(p_b - prev_p))
                            prev_p = p_b
                            if s >= 3:
                                chord = float(np.linalg.norm(p_b - p_a))
                                if chord < med_edge * 4.5 and arc_acc > max(med_edge * 5.0, chord * 2.6):
                                    jump_to = idx_l + s
                        pruned_loop.append(loop_nodes[idx_l])
                        if jump_to != -1:
                            idx_l = jump_to
                        else:
                            idx_l += 1
                    if 3 <= len(pruned_loop) < len(loop_nodes):
                        loop_nodes = pruned_loop
                    else:
                        break

            L0 = len(loop_nodes)
            if L0 < 3:
                return

            pts0 = cur_verts[np.array(loop_nodes, dtype=np.int64)]
            seg_d = np.linalg.norm(np.roll(pts0, -1, axis=0) - pts0, axis=1)
            if float(seg_d.max()) > bbox_diag * 0.24:
                return

            ring_ids = []
            ring_pts_list = []
            ring_cols_list = []
            sub_verts_new = []
            sub_cols_new = []

            for i in range(L0):
                u_id = loop_nodes[i]
                v_id = loop_nodes[(i + 1) % L0]
                p_u = cur_verts[u_id]
                p_v = cur_verts[v_id]
                c_u = (
                    cur_colors[u_id].astype(np.float64)
                    if (colors_list is not None and cur_colors is not None)
                    else None
                )
                c_v = (
                    cur_colors[v_id].astype(np.float64)
                    if (colors_list is not None and cur_colors is not None)
                    else None
                )
                ring_ids.append(u_id)
                ring_pts_list.append(p_u)
                if c_u is not None:
                    ring_cols_list.append(c_u)

                d_uv = float(np.linalg.norm(p_v - p_u))
                if med_edge > 1e-6 and d_uv > med_edge * 2.2:
                    n_sub = min(48, int(math.ceil(d_uv / med_edge)) - 1)
                    for s in range(1, n_sub + 1):
                        t = float(s) / float(n_sub + 1)
                        sp = (1.0 - t) * p_u + t * p_v
                        sub_verts_new.append(sp)
                        ring_ids.append(next_vid)
                        next_vid += 1
                        ring_pts_list.append(sp)
                        if c_u is not None and c_v is not None:
                            sc = (1.0 - t) * c_u + t * c_v
                            sub_cols_new.append(sc)
                            ring_cols_list.append(sc)

            if sub_verts_new:
                verts_list.append(np.array(sub_verts_new, dtype=np.float64))
                if colors_list is not None and sub_cols_new and cur_colors is not None:
                    sc_arr = np.array(sub_cols_new, dtype=np.float64)
                    if np.issubdtype(cur_colors.dtype, np.integer):
                        sc_arr = np.clip(np.round(sc_arr), 0, 255).astype(cur_colors.dtype)
                    else:
                        sc_arr = sc_arr.astype(cur_colors.dtype)
                    colors_list.append(sc_arr)

            curr_pts = np.array(ring_pts_list, dtype=np.float64)
            seg_w = np.linalg.norm(np.roll(curr_pts, -1, axis=0) - curr_pts, axis=1, keepdims=True) + 1e-7
            centroid = (0.5 * (curr_pts + np.roll(curr_pts, -1, axis=0)) * seg_w).sum(axis=0, keepdims=True) / seg_w.sum()
            curr_cols = (
                np.array(ring_cols_list, dtype=np.float64)
                if ring_cols_list
                else None
            )
            c_mean = curr_cols.mean(axis=0, keepdims=True) if curr_cols is not None else None
            curr_ids = list(ring_ids)

            # 2-to-1 Advancing-Front Ring Reduction:
            # Halve ring vertex count (M -> M // 2) at each concentric step until M <= 8.
            # Every vertex in the cap has degree <= 8 (never creates a high-degree starburst hub).
            while len(curr_ids) > 8:
                M = len(curr_ids)
                next_M = max(4, M // 2)

                # Smooth contour before halving
                sm_pts = curr_pts.copy()
                sm_cols = curr_cols.copy() if curr_cols is not None else None
                for _ in range(2):
                    sm_pts = (
                        0.25 * np.roll(sm_pts, 1, axis=0)
                        + 0.50 * sm_pts
                        + 0.25 * np.roll(sm_pts, -1, axis=0)
                    )
                    if sm_cols is not None:
                        sm_cols = (
                            0.25 * np.roll(sm_cols, 1, axis=0)
                            + 0.50 * sm_cols
                            + 0.25 * np.roll(sm_cols, -1, axis=0)
                        )

                steps_rem = max(1.0, math.log2(M / 4.0))
                alpha = min(0.42, max(0.18, 1.0 / (steps_rem + 0.8)))

                next_pts = np.empty((next_M, 3), dtype=np.float64)
                next_cols = (
                    np.empty((next_M, curr_cols.shape[1]), dtype=np.float64)
                    if (curr_cols is not None and sm_cols is not None)
                    else None
                )
                next_ids = []

                for j in range(next_M):
                    src0 = (2 * j) % M
                    src1 = (2 * j + 1) % M
                    mid_pt = 0.5 * (sm_pts[src0] + sm_pts[src1])
                    next_pts[j] = (1.0 - alpha) * mid_pt + alpha * centroid[0]
                    if next_cols is not None and sm_cols is not None and c_mean is not None:
                        mid_col = 0.5 * (sm_cols[src0] + sm_cols[src1])
                        next_cols[j] = (1.0 - alpha) * mid_col + alpha * c_mean[0]
                    next_ids.append(next_vid)
                    next_vid += 1

                verts_list.append(next_pts)
                if colors_list is not None and next_cols is not None and cur_colors is not None:
                    out_cols = next_cols
                    if np.issubdtype(cur_colors.dtype, np.integer):
                        out_cols = np.clip(np.round(out_cols), 0, 255).astype(cur_colors.dtype)
                    else:
                        out_cols = out_cols.astype(cur_colors.dtype)
                    colors_list.append(out_cols)

                strip_tris = []
                for j in range(next_M):
                    nj = (j + 1) % next_M
                    b0 = next_ids[j]
                    b1 = next_ids[nj]
                    if j < next_M - 1 or M == 2 * next_M:
                        i0 = 2 * j
                        i1 = (2 * j + 1) % M
                        i2 = (2 * j + 2) % M
                        u0, u1, u2 = curr_ids[i0], curr_ids[i1], curr_ids[i2]
                        strip_tris.append([u0, u1, b0])
                        strip_tris.append([u1, b1, b0])
                        strip_tris.append([u1, u2, b1])
                    else:
                        # Odd remainder: 3 outer edges -> 1 inner edge (4 triangles)
                        i0 = 2 * j
                        i1 = (2 * j + 1) % M
                        i2 = (2 * j + 2) % M
                        i3 = 0
                        u0, u1, u2, u3 = curr_ids[i0], curr_ids[i1], curr_ids[i2], curr_ids[i3]
                        strip_tris.append([u0, u1, b0])
                        strip_tris.append([u1, u2, b0])
                        strip_tris.append([u2, b1, b0])
                        strip_tris.append([u2, u3, b1])

                new_faces.append(np.array(strip_tris, dtype=np.int64))
                curr_ids = next_ids
                curr_pts = next_pts
                curr_cols = next_cols

            # Close remaining small core (len(curr_ids) <= 8) with a tiny center vertex (degree <= 8)
            K = len(curr_ids)
            verts_list.append(centroid)
            if colors_list is not None and c_mean is not None and cur_colors is not None:
                c_out = c_mean
                if np.issubdtype(cur_colors.dtype, np.integer):
                    c_out = np.clip(np.round(c_out), 0, 255).astype(cur_colors.dtype)
                else:
                    c_out = c_out.astype(cur_colors.dtype)
                colors_list.append(c_out)
            center_vid = next_vid
            next_vid += 1

            fan = np.array(
                [[curr_ids[i], curr_ids[(i + 1) % K], center_vid] for i in range(K)],
                dtype=np.int64,
            )
            new_faces.append(fan)
            loops_in_pass += 1

        open_chains = []

        for e_start in list(boundary_edges):
            if e_start in used_edges:
                continue
            u0, v0 = e_start
            if (v0, u0) in pref_dir:
                u0, v0 = v0, u0

            used_edges.add(e_start)
            path_nodes = [u0, v0]
            visited_idx = {u0: 0, v0: 1}
            spur_backtracks = 0

            while len(path_nodes) > 0:
                curr = path_nodes[-1]
                candidates = []
                for nb in undir_adj[curr]:
                    ek = (min(curr, nb), max(curr, nb))
                    if ek not in used_edges:
                        candidates.append((nb, ek))

                if not candidates:
                    # Only backtrack if an ancestor within <= 24 steps still has unused boundary edges
                    branch_k = -1
                    for k in range(len(path_nodes) - 2, -1, -1):
                        nk = path_nodes[k]
                        if any((min(nk, w), max(nk, w)) not in used_edges for w in undir_adj[nk]):
                            branch_k = k
                            break
                    if branch_k != -1 and (len(path_nodes) - 1 - branch_k) <= 24 and spur_backtracks <= 32:
                        while len(path_nodes) > branch_k + 1:
                            dead = path_nodes.pop()
                            visited_idx.pop(dead, None)
                        spur_backtracks += 1
                        continue
                    break

                spur_backtracks = 0
                has_unvisited_cont = any(
                    nb not in visited_idx
                    and any((min(nb, w), max(nb, w)) not in used_edges and (min(nb, w), max(nb, w)) != ek for w in undir_adj[nb])
                    for nb, ek in candidates
                )

                # Prefer major cycle closure, or continuing unvisited preferred edge
                chosen_nb, chosen_ek = candidates[0]
                best_sc = -1e9
                for nb, ek in candidates:
                    sc = 1.0
                    if nb in visited_idx and len(path_nodes) - visited_idx[nb] >= 3:
                        cyc_len = len(path_nodes) - visited_idx[nb]
                        if visited_idx[nb] == 0 or cyc_len >= 16 or not has_unvisited_cont:
                            sc = 200.0
                        else:
                            sc = 5.0
                    else:
                        rem_d = sum(1 for w in undir_adj[nb] if (min(nb, w), max(nb, w)) not in used_edges and (min(nb, w), max(nb, w)) != ek)
                        if rem_d >= 1 and (curr, nb) in pref_dir:
                            sc = 60.0
                        elif rem_d >= 1:
                            sc = 40.0
                        elif (curr, nb) in pref_dir:
                            sc = 15.0
                    if sc > best_sc:
                        best_sc = sc
                        chosen_nb, chosen_ek = nb, ek

                used_edges.add(chosen_ek)
                if chosen_nb in visited_idx:
                    loop_start = visited_idx[chosen_nb]
                    loop = path_nodes[loop_start:]
                    if len(loop) >= 3:
                        _seal_loop(loop)
                    for removed in path_nodes[loop_start + 1 :]:
                        visited_idx.pop(removed, None)
                    path_nodes = path_nodes[: loop_start + 1]
                    if len(loop) >= 12 and loop_start <= 16:
                        rem_at_start = sum(
                            1 for w in undir_adj[path_nodes[-1]]
                            if (min(path_nodes[-1], w), max(path_nodes[-1], w)) not in used_edges
                        )
                        if rem_at_start == 0:
                            path_nodes = []
                            break
                else:
                    visited_idx[chosen_nb] = len(path_nodes)
                    path_nodes.append(chosen_nb)

            # Extend backward from u0 if an open chain remains
            if len(path_nodes) >= 2:
                back_nodes = []
                curr_back = u0
                visited_back = set(path_nodes)
                while True:
                    back_cands = []
                    for nb in undir_adj[curr_back]:
                        ek = (min(curr_back, nb), max(curr_back, nb))
                        if ek not in used_edges and nb not in visited_back:
                            back_cands.append((nb, ek))
                    if not back_cands:
                        break
                    chosen_nb, chosen_ek = back_cands[0]
                    for nb, ek in back_cands:
                        if (nb, curr_back) in pref_dir:
                            chosen_nb, chosen_ek = nb, ek
                            break
                    used_edges.add(chosen_ek)
                    back_nodes.append(chosen_nb)
                    visited_back.add(chosen_nb)
                    curr_back = chosen_nb

                full_chain = list(reversed(back_nodes)) + path_nodes
                if len(full_chain) >= 3:
                    open_chains.append(full_chain)

        # Multi-Chain Perimeter Stitcher: stitch open boundary chains around large base holes
        if open_chains:
            open_chains.sort(key=len, reverse=True)
            chain_used = [False] * len(open_chains)

            def _chain_perim(ch):
                pts = cur_verts[np.array(ch, dtype=np.int64)]
                return float(np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum())

            for i in range(len(open_chains)):
                if chain_used[i] or len(open_chains[i]) < 3:
                    continue
                chain_used[i] = True
                stitched = list(open_chains[i])

                while True:
                    head = stitched[0]
                    tail = stitched[-1]
                    self_gap = float(np.linalg.norm(cur_verts[tail] - cur_verts[head]))
                    cur_perim = _chain_perim(stitched)

                    best_idx = -1
                    best_rev = False
                    best_prepend = False
                    best_dist = float("inf")

                    for j in range(len(open_chains)):
                        if chain_used[j] or len(open_chains[j]) < 3:
                            continue
                        cj = open_chains[j]
                        s_j, e_j = cj[0], cj[-1]
                        d1 = float(np.linalg.norm(cur_verts[tail] - cur_verts[s_j]))
                        if d1 < best_dist:
                            best_dist, best_idx, best_rev, best_prepend = d1, j, False, False
                        d2 = float(np.linalg.norm(cur_verts[tail] - cur_verts[e_j])) * 1.05
                        if d2 < best_dist:
                            best_dist, best_idx, best_rev, best_prepend = d2, j, True, False
                        d3 = float(np.linalg.norm(cur_verts[e_j] - cur_verts[head]))
                        if d3 < best_dist:
                            best_dist, best_idx, best_rev, best_prepend = d3, j, False, True
                        d4 = float(np.linalg.norm(cur_verts[s_j] - cur_verts[head])) * 1.05
                        if d4 < best_dist:
                            best_dist, best_idx, best_rev, best_prepend = d4, j, True, True

                    if len(stitched) >= 8 and cur_perim > 1e-5 and self_gap < cur_perim * 0.35 and self_gap < bbox_diag * 0.22 and self_gap <= best_dist * 1.15:
                        break

                    if best_idx != -1 and best_dist < bbox_diag * 0.16:
                        chain_used[best_idx] = True
                        nxt = list(open_chains[best_idx])
                        if best_rev:
                            nxt.reverse()
                        if best_prepend:
                            stitched = nxt + stitched
                        else:
                            stitched = stitched + nxt
                    else:
                        break

                if len(stitched) >= 3:
                    c_perim = _chain_perim(stitched)
                    g_dist = float(np.linalg.norm(cur_verts[stitched[0]] - cur_verts[stitched[-1]]))
                    if c_perim > 1e-6 and g_dist < min(c_perim * 0.45, bbox_diag * 0.22):
                        _seal_loop(stitched)

        faces = np.vstack(new_faces)
        total_loops_closed += loops_in_pass
        if loops_in_pass == 0:
            continue

    final_verts = np.vstack(verts_list)
    final_colors = np.vstack(colors_list) if colors_list is not None else None
    return final_verts, faces, final_colors, total_loops_closed


def _make_obj_watertight(path):
    """
    Close all open boundary loops in a Wavefront .obj file while preserving
    existing UV coordinates (vt), normals (vn), mtllib, and usemtl directives.
    Uses spatial 3D vertex welding so UV seams never split hole boundaries, and
    seals holes with 2-to-1 Advancing-Front Ring Reduction (max vertex degree <= 8)
    and a single representative UV per cap to prevent UV texture stretching or
    frontend spoke-filter deletion.
    """
    import math
    import numpy as np
    from collections import Counter, defaultdict

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

    bbox_min = verts_arr.min(axis=0)
    bbox_max = verts_arr.max(axis=0)
    bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
    if bbox_diag < 1e-6:
        bbox_diag = 1.0

    # Weld vertices by 3D spatial position so UV seams do not split hole loops
    quant = np.round((verts_arr - bbox_min) * (25000.0 / bbox_diag)).astype(np.int64)
    _, u_idx, weld_inv = np.unique(quant, axis=0, return_index=True, return_inverse=True)
    rep_vid = u_idx[weld_inv]

    # Propagate UV mapping to welded representative vertices
    for v_orig, vt_idx in list(face_v_to_vt.items()):
        w_rep = int(rep_vid[v_orig])
        if w_rep not in face_v_to_vt:
            face_v_to_vt[w_rep] = vt_idx

    w_faces = rep_vid[faces_arr]
    valid_wf = (
        (w_faces[:, 0] != w_faces[:, 1])
        & (w_faces[:, 1] != w_faces[:, 2])
        & (w_faces[:, 2] != w_faces[:, 0])
    )
    w_faces = w_faces[valid_wf]
    if len(w_faces) == 0:
        return False

    h0 = w_faces[:, [0, 1]]
    h1 = w_faces[:, [1, 2]]
    h2 = w_faces[:, [2, 0]]
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
        print("[Watertight] OBJ-сітка " + str(path.name) + " вже герметична.", flush=True)
        return True

    has_vt = len(vts) > 0 and len(face_v_to_vt) > 0
    vts_arr = np.asarray(vts, dtype=np.float64) if has_vt else None
    next_v_idx = orig_v_count + 1  # 1-based OBJ vertex index
    next_vt_idx = len(vts) + 1     # 1-based OBJ vt index
    append_v_lines = []
    append_vt_lines = []
    append_f_lines = []
    loops_closed = 0
    used_edges = set()

    # Prune degree-1 dead-end spurs at branch vertices
    for _prune_it in range(8):
        pruned_any = False
        for u, nbs in list(undir_adj.items()):
            alive = [nb for nb in nbs if (min(u, nb), max(u, nb)) not in used_edges]
            if len(alive) == 1:
                nb = alive[0]
                nb_alive = [w for w in undir_adj[nb] if (min(nb, w), max(nb, w)) not in used_edges]
                if len(nb_alive) >= 3:
                    used_edges.add((min(u, nb), max(u, nb)))
                    pruned_any = True
        if not pruned_any:
            break

    def _emit_obj_tri(idx_a, vt_a, idx_b, vt_b, idx_c, vt_c):
        # idx_a, idx_b, idx_c are 1-based OBJ vertex indices
        if has_vt:
            append_f_lines.append("f %d/%d %d/%d %d/%d" % (idx_a, vt_a, idx_b, vt_b, idx_c, vt_c))
        else:
            append_f_lines.append("f %d %d %d" % (idx_a, idx_b, idx_c))

    def _seal_obj_loop(raw_loop):
        nonlocal next_v_idx, next_vt_idx, loops_closed
        loop = []
        for nid in raw_loop:
            nid_i = int(nid)
            if loop and loop[-1] == nid_i:
                continue
            if len(loop) >= 2 and loop[-2] == nid_i:
                loop.pop()
                continue
            loop.append(nid_i)
        while len(loop) > 2 and loop[0] == loop[-1]:
            loop.pop()

        L = len(loop)
        if L < 3:
            return

        fwd_score = sum(1 for i in range(L) if (loop[i], loop[(i + 1) % L]) in pref_dir)
        rev_score = sum(1 for i in range(L) if (loop[(i + 1) % L], loop[i]) in pref_dir)
        if rev_score > fwd_score:
            loop = list(reversed(loop))

        # Pick ONE representative UV coordinate from the boundary to avoid UV island stretching
        rep_vt_1based = 1
        if has_vt and vts_arr is not None:
            for n in loop:
                if n in face_v_to_vt and 0 <= face_v_to_vt[n] < len(vts_arr):
                    uv = vts_arr[face_v_to_vt[n]]
                    append_vt_lines.append("vt %.6f %.6f" % (uv[0], uv[1]))
                    rep_vt_1based = next_vt_idx
                    next_vt_idx += 1
                    break

        if L == 3:
            v1, v2, v3 = loop[0] + 1, loop[1] + 1, loop[2] + 1
            t1 = face_v_to_vt.get(loop[0], 0) + 1 if has_vt else 1
            t2 = face_v_to_vt.get(loop[1], 0) + 1 if has_vt else 1
            t3 = face_v_to_vt.get(loop[2], 0) + 1 if has_vt else 1
            _emit_obj_tri(v1, t1, v2, t2, v3, t3)
            loops_closed += 1
            return

        if L == 4:
            v0, v1, v2, v3 = [n + 1 for n in loop]
            t0 = face_v_to_vt.get(loop[0], 0) + 1 if has_vt else 1
            t1 = face_v_to_vt.get(loop[1], 0) + 1 if has_vt else 1
            t2 = face_v_to_vt.get(loop[2], 0) + 1 if has_vt else 1
            t3 = face_v_to_vt.get(loop[3], 0) + 1 if has_vt else 1
            _emit_obj_tri(v0, t0, v1, t1, v2, t2)
            _emit_obj_tri(v0, t0, v2, t2, v3, t3)
            loops_closed += 1
            return

        # Subdivide any long bridged gap (> 2.2x median loop edge) so all sides of the hole
        # have uniform vertex density before 2-to-1 ring reduction
        pts0 = verts_arr[np.array(loop, dtype=np.int64)]
        seg_d = np.linalg.norm(np.roll(pts0, -1, axis=0) - pts0, axis=1)
        valid_d = seg_d[seg_d > 1e-6]
        med_edge = float(np.median(valid_d)) if len(valid_d) > 0 else bbox_diag * 0.015

        curr_ring = []
        ring_pts_list = []
        for i in range(L):
            u_id = loop[i]
            v_id = loop[(i + 1) % L]
            p_u = verts_arr[u_id]
            p_v = verts_arr[v_id]
            vt_u = (face_v_to_vt.get(u_id, 0) + 1) if has_vt else 1
            curr_ring.append((u_id + 1, vt_u))
            ring_pts_list.append(p_u)

            d_uv = float(np.linalg.norm(p_v - p_u))
            if med_edge > 1e-6 and d_uv > med_edge * 2.2:
                n_sub = min(48, int(math.ceil(d_uv / med_edge)) - 1)
                for s in range(1, n_sub + 1):
                    t = float(s) / float(n_sub + 1)
                    sp = (1.0 - t) * p_u + t * p_v
                    append_v_lines.append("v %.6f %.6f %.6f" % (sp[0], sp[1], sp[2]))
                    curr_ring.append((next_v_idx, rep_vt_1based))
                    next_v_idx += 1
                    ring_pts_list.append(sp)

        curr_pts = np.array(ring_pts_list, dtype=np.float64)
        seg_w = np.linalg.norm(np.roll(curr_pts, -1, axis=0) - curr_pts, axis=1, keepdims=True) + 1e-7
        centroid = (0.5 * (curr_pts + np.roll(curr_pts, -1, axis=0)) * seg_w).sum(axis=0) / float(seg_w.sum())

        while len(curr_ring) > 8:
            M = len(curr_ring)
            next_M = max(4, M // 2)
            sm_pts = curr_pts.copy()
            for _ in range(2):
                sm_pts = (
                    0.25 * np.roll(sm_pts, 1, axis=0)
                    + 0.50 * sm_pts
                    + 0.25 * np.roll(sm_pts, -1, axis=0)
                )
            steps_rem = max(1.0, math.log2(M / 4.0))
            alpha = min(0.42, max(0.18, 1.0 / (steps_rem + 0.8)))

            next_pts = np.empty((next_M, 3), dtype=np.float64)
            next_ring = []
            for j in range(next_M):
                src0 = (2 * j) % M
                src1 = (2 * j + 1) % M
                mid_pt = 0.5 * (sm_pts[src0] + sm_pts[src1])
                pt = (1.0 - alpha) * mid_pt + alpha * centroid
                next_pts[j] = pt
                append_v_lines.append("v %.6f %.6f %.6f" % (pt[0], pt[1], pt[2]))
                next_ring.append((next_v_idx, rep_vt_1based))
                next_v_idx += 1

            for j in range(next_M):
                nj = (j + 1) % next_M
                b0_v, b0_t = next_ring[j]
                b1_v, b1_t = next_ring[nj]
                if j < next_M - 1 or M == 2 * next_M:
                    i0 = 2 * j
                    i1 = (2 * j + 1) % M
                    i2 = (2 * j + 2) % M
                    u0_v, u0_t = curr_ring[i0]
                    u1_v, u1_t = curr_ring[i1]
                    u2_v, u2_t = curr_ring[i2]
                    _emit_obj_tri(u0_v, u0_t, u1_v, u1_t, b0_v, b0_t)
                    _emit_obj_tri(u1_v, u1_t, b1_v, b1_t, b0_v, b0_t)
                    _emit_obj_tri(u1_v, u1_t, u2_v, u2_t, b1_v, b1_t)
                else:
                    i0 = 2 * j
                    i1 = (2 * j + 1) % M
                    i2 = (2 * j + 2) % M
                    i3 = 0
                    u0_v, u0_t = curr_ring[i0]
                    u1_v, u1_t = curr_ring[i1]
                    u2_v, u2_t = curr_ring[i2]
                    u3_v, u3_t = curr_ring[i3]
                    _emit_obj_tri(u0_v, u0_t, u1_v, u1_t, b0_v, b0_t)
                    _emit_obj_tri(u1_v, u1_t, u2_v, u2_t, b0_v, b0_t)
                    _emit_obj_tri(u2_v, u2_t, b1_v, b1_t, b0_v, b0_t)
                    _emit_obj_tri(u2_v, u2_t, u3_v, u3_t, b1_v, b1_t)

            curr_ring = next_ring
            curr_pts = next_pts

        # Close remaining <= 8 core
        K = len(curr_ring)
        append_v_lines.append("v %.6f %.6f %.6f" % (centroid[0], centroid[1], centroid[2]))
        c_v_idx = next_v_idx
        next_v_idx += 1
        for i in range(K):
            a_v, a_t = curr_ring[i]
            b_v, b_t = curr_ring[(i + 1) % K]
            _emit_obj_tri(a_v, a_t, b_v, b_t, c_v_idx, rep_vt_1based)

        loops_closed += 1

    open_chains = []

    for e_start in list(boundary_edges):
        if e_start in used_edges:
            continue
        u0, v0 = e_start
        if (v0, u0) in pref_dir:
            u0, v0 = v0, u0
        used_edges.add(e_start)
        path_nodes = [u0, v0]
        visited_idx = {u0: 0, v0: 1}
        spur_backtracks = 0

        while len(path_nodes) > 0:
            curr = path_nodes[-1]
            candidates = [
                (nb, (min(curr, nb), max(curr, nb)))
                for nb in undir_adj[curr]
                if (min(curr, nb), max(curr, nb)) not in used_edges
            ]
            if not candidates:
                branch_k = -1
                for k in range(len(path_nodes) - 2, -1, -1):
                    nk = path_nodes[k]
                    if any((min(nk, w), max(nk, w)) not in used_edges for w in undir_adj[nk]):
                        branch_k = k
                        break
                if branch_k != -1 and (len(path_nodes) - 1 - branch_k) <= 24 and spur_backtracks <= 32:
                    while len(path_nodes) > branch_k + 1:
                        dead = path_nodes.pop()
                        visited_idx.pop(dead, None)
                    spur_backtracks += 1
                    continue
                break

            spur_backtracks = 0
            has_unvisited_cont = any(
                nb not in visited_idx
                and any((min(nb, w), max(nb, w)) not in used_edges and (min(nb, w), max(nb, w)) != ek for w in undir_adj[nb])
                for nb, ek in candidates
            )

            chosen_nb, chosen_ek = candidates[0]
            best_sc = -1e9
            for nb, ek in candidates:
                sc = 1.0
                if nb in visited_idx and len(path_nodes) - visited_idx[nb] >= 3:
                    cyc_len = len(path_nodes) - visited_idx[nb]
                    if visited_idx[nb] == 0 or cyc_len >= 16 or not has_unvisited_cont:
                        sc = 200.0
                    else:
                        sc = 5.0
                else:
                    rem_d = sum(1 for w in undir_adj[nb] if (min(nb, w), max(nb, w)) not in used_edges and (min(nb, w), max(nb, w)) != ek)
                    if rem_d >= 1 and (curr, nb) in pref_dir:
                        sc = 60.0
                    elif rem_d >= 1:
                        sc = 40.0
                    elif (curr, nb) in pref_dir:
                        sc = 15.0
                if sc > best_sc:
                    best_sc = sc
                    chosen_nb, chosen_ek = nb, ek

            used_edges.add(chosen_ek)
            if chosen_nb in visited_idx:
                loop_start = visited_idx[chosen_nb]
                loop = path_nodes[loop_start:]
                if len(loop) >= 3:
                    _seal_obj_loop(loop)
                for removed in path_nodes[loop_start + 1 :]:
                    visited_idx.pop(removed, None)
                path_nodes = path_nodes[: loop_start + 1]
                if len(loop) >= 12 and loop_start <= 16:
                    rem_at_start = sum(
                        1 for w in undir_adj[path_nodes[-1]]
                        if (min(path_nodes[-1], w), max(path_nodes[-1], w)) not in used_edges
                    )
                    if rem_at_start == 0:
                        path_nodes = []
                        break
            else:
                visited_idx[chosen_nb] = len(path_nodes)
                path_nodes.append(chosen_nb)

        if len(path_nodes) >= 2:
            back_nodes = []
            curr_back = u0
            visited_back = set(path_nodes)
            while True:
                back_cands = [
                    (nb, (min(curr_back, nb), max(curr_back, nb)))
                    for nb in undir_adj[curr_back]
                    if (min(curr_back, nb), max(curr_back, nb)) not in used_edges and nb not in visited_back
                ]
                if not back_cands:
                    break
                chosen_nb, chosen_ek = back_cands[0]
                for nb, ek in back_cands:
                    if (nb, curr_back) in pref_dir:
                        chosen_nb, chosen_ek = nb, ek
                        break
                used_edges.add(chosen_ek)
                back_nodes.append(chosen_nb)
                visited_back.add(chosen_nb)
                curr_back = chosen_nb

            full_chain = list(reversed(back_nodes)) + path_nodes
            if len(full_chain) >= 3:
                open_chains.append(full_chain)

    # Multi-Chain Perimeter Stitcher for OBJ
    if open_chains:
        open_chains.sort(key=len, reverse=True)
        chain_used = [False] * len(open_chains)

        def _obj_chain_perim(ch):
            pts = verts_arr[np.array(ch, dtype=np.int64)]
            return float(np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum())

        for i in range(len(open_chains)):
            if chain_used[i] or len(open_chains[i]) < 3:
                continue
            chain_used[i] = True
            stitched = list(open_chains[i])

            while True:
                head = stitched[0]
                tail = stitched[-1]
                self_gap = float(np.linalg.norm(verts_arr[tail] - verts_arr[head]))
                cur_perim = _obj_chain_perim(stitched)

                best_idx = -1
                best_rev = False
                best_prepend = False
                best_dist = float("inf")

                for j in range(len(open_chains)):
                    if chain_used[j] or len(open_chains[j]) < 3:
                        continue
                    cj = open_chains[j]
                    s_j, e_j = cj[0], cj[-1]
                    d1 = float(np.linalg.norm(verts_arr[tail] - verts_arr[s_j]))
                    if d1 < best_dist:
                        best_dist, best_idx, best_rev, best_prepend = d1, j, False, False
                    d2 = float(np.linalg.norm(verts_arr[tail] - verts_arr[e_j])) * 1.05
                    if d2 < best_dist:
                        best_dist, best_idx, best_rev, best_prepend = d2, j, True, False
                    d3 = float(np.linalg.norm(verts_arr[e_j] - verts_arr[head]))
                    if d3 < best_dist:
                        best_dist, best_idx, best_rev, best_prepend = d3, j, False, True
                    d4 = float(np.linalg.norm(verts_arr[s_j] - verts_arr[head])) * 1.05
                    if d4 < best_dist:
                        best_dist, best_idx, best_rev, best_prepend = d4, j, True, True

                if len(stitched) >= 8 and cur_perim > 1e-5 and self_gap < cur_perim * 0.35 and self_gap < bbox_diag * 0.22 and self_gap <= best_dist * 1.15:
                    break

                if best_idx != -1 and best_dist < bbox_diag * 0.16:
                    chain_used[best_idx] = True
                    nxt = list(open_chains[best_idx])
                    if best_rev:
                        nxt.reverse()
                    if best_prepend:
                        stitched = nxt + stitched
                    else:
                        stitched = stitched + nxt
                else:
                    break

            if len(stitched) >= 3:
                c_perim = _obj_chain_perim(stitched)
                g_dist = float(np.linalg.norm(verts_arr[stitched[0]] - verts_arr[stitched[-1]]))
                if c_perim > 1e-6 and g_dist < min(c_perim * 0.45, bbox_diag * 0.22):
                    _seal_obj_loop(stitched)

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


def _repair_open3d_mesh_in_place(mesh, label="mesh"):
    """
    Clean non-manifold edges and close all open boundary loops on an Open3D TriangleMesh object in memory.
    """
    import numpy as np
    import open3d as o3d

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

    repaired_verts, repaired_faces, repaired_colors, loops_closed = _close_mesh_boundaries_numpy(
        verts, faces, colors
    )

    out_mesh = o3d.geometry.TriangleMesh()
    out_mesh.vertices = o3d.utility.Vector3dVector(repaired_verts)
    out_mesh.triangles = o3d.utility.Vector3iVector(repaired_faces.astype(np.int32))
    if repaired_colors is not None and len(repaired_colors) == len(repaired_verts):
        out_mesh.vertex_colors = o3d.utility.Vector3dVector(repaired_colors)
    out_mesh.remove_duplicated_triangles()
    out_mesh.remove_degenerate_triangles()
    out_mesh.remove_unreferenced_vertices()
    out_mesh.compute_vertex_normals()

    print(
        "[Watertight] Готово (%s): зашито %d відкритих контурів -> %d V, %d F."
        % (label, loops_closed, len(out_mesh.vertices), len(out_mesh.triangles)),
        flush=True,
    )
    return out_mesh, loops_closed


_IN_WATERTIGHT_WRITE = False


def make_mesh_watertight(mesh_file):
    """
    Close all open boundary loops (holes) in a triangle mesh (.ply or .obj)
    while preserving vertex colors (for .ply) or UV textures (for .obj).
    Uses Open3D + NumPy (installed natively in 2d-sugar environment) with a
    plyfile fallback, requiring zero external dependencies.
    """
    global _IN_WATERTIGHT_WRITE
    if _IN_WATERTIGHT_WRITE:
        return False

    path = Path(str(mesh_file))
    if not path.is_file():
        return False

    suffix = path.suffix.lower()
    if suffix == ".obj":
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

        # Strategy 1: Open3D (standard in 2d-sugar environment)
        try:
            import open3d as o3d

            mesh = o3d.io.read_triangle_mesh(str(path))
            if len(mesh.triangles) == 0:
                return False

            out_mesh, _ = _repair_open3d_mesh_in_place(mesh, label=path.name)
            o3d.io.write_triangle_mesh(str(path), out_mesh)
            return True
        except ImportError:
            pass

        # Strategy 2: plyfile fallback
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

        repaired_verts, repaired_faces, repaired_colors, loops_closed = _close_mesh_boundaries_numpy(
            verts, faces, colors
        )

        if has_rgb and repaired_colors is not None:
            v_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
            v_arr = np.empty(len(repaired_verts), dtype=v_dtype)
            v_arr["x"] = repaired_verts[:, 0].astype(np.float32)
            v_arr["y"] = repaired_verts[:, 1].astype(np.float32)
            v_arr["z"] = repaired_verts[:, 2].astype(np.float32)
            v_arr["red"] = repaired_colors[:, 0].astype(np.uint8)
            v_arr["green"] = repaired_colors[:, 1].astype(np.uint8)
            v_arr["blue"] = repaired_colors[:, 2].astype(np.uint8)
        else:
            v_dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
            v_arr = np.empty(len(repaired_verts), dtype=v_dtype)
            v_arr["x"] = repaired_verts[:, 0].astype(np.float32)
            v_arr["y"] = repaired_verts[:, 1].astype(np.float32)
            v_arr["z"] = repaired_verts[:, 2].astype(np.float32)

        f_dtype = [(f_prop, "i4", (3,))]
        f_arr = np.empty(len(repaired_faces), dtype=f_dtype)
        f_arr[f_prop] = repaired_faces.astype(np.int32)

        PlyData([PlyElement.describe(v_arr, "vertex"), PlyElement.describe(f_arr, "face")], text=False).write(
            str(path)
        )
        print(
            "[Watertight] Готово (%s, plyfile): зашито %d дірок -> %d V, %d F."
            % (path.name, loops_closed, len(repaired_verts), len(repaired_faces)),
            flush=True,
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
def _patch_open3d_io(o3d_mod):
    if not _watertight_flag:
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
                mesh, _ = _repair_open3d_mesh_in_place(mesh, label=os.path.basename(fname_str))
            except Exception:
                pass
        return orig_write(filename, mesh, *w_args, **w_kwargs)

    io_mod.write_triangle_mesh = patched_write_triangle_mesh


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

            ctx = _get_visual_hull_context()
            req_points = num_generated_points
            if ctx and isinstance(num_generated_points, int) and num_generated_points > 0:
                req_points = min(num_generated_points * 2, 180000)

            xyz, rgb, normals = orig_gen(cam_infos, cam_intrinsics, req_points, image_depth_scale_map, *g_args, **g_kwargs)

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

                if ctx and len(xyz) > 100:
                    vis_cnt, fg_cnt = _project_points_to_visual_hull(xyz, ctx)
                    fg_ratio = fg_cnt.astype(np.float64) / np.maximum(vis_cnt, 1).astype(np.float64)
                    rgb_max = rgb.max(axis=1) if rgb.ndim == 2 and rgb.shape[1] >= 3 else np.ones(len(xyz))
                    pure_black_bg = rgb_max <= (4.0 if rgb_max.max() > 1.5 else 0.016)
                    keep = (vis_cnt >= 3) & (fg_ratio >= 0.55) & (~(pure_black_bg & (fg_ratio < 0.85)))
                    if int(keep.sum()) < 300:
                        keep = (vis_cnt >= 2) & (fg_ratio >= 0.45)
                    if int(keep.sum()) >= 100:
                        before_n = len(xyz)
                        xyz = xyz[keep]
                        rgb = rgb[keep]
                        normals = normals[keep]
                        if isinstance(num_generated_points, int) and len(xyz) > num_generated_points > 0:
                            step_idx = np.linspace(0, len(xyz) - 1, num_generated_points, dtype=np.int64)
                            xyz = xyz[step_idx]
                            rgb = rgb[step_idx]
                            normals = normals[step_idx]
                        print(
                            "[VisualHull] Видалено чорні фонові гаусіани під час генерації: %d -> %d (відсіяно %d поза об'єктом)."
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


# ==============================================================================
# Visual-Hull guards for training (Gaussian pruning) and TSDF mesh extraction (depth masking)
# ==============================================================================
_FULL_MASKS_CACHE = None
_TORCH_VH_CACHE = None


def _get_all_masks_by_stem():
    """All foreground masks ($DATASET/masks/*.png) keyed by stem (not subsampled)."""
    global _FULL_MASKS_CACHE
    if _FULL_MASKS_CACHE is not None:
        return _FULL_MASKS_CACHE
    _FULL_MASKS_CACHE = {}
    if not _mask_fg_flag:
        return _FULL_MASKS_CACHE
    ds_dir = _resolve_dataset_dir()
    if ds_dir is None or not (ds_dir / "masks").is_dir():
        return _FULL_MASKS_CACHE
    try:
        import numpy as np
        from PIL import Image

        for mf in sorted((ds_dir / "masks").glob("*.png")):
            try:
                _FULL_MASKS_CACHE[mf.stem] = np.asarray(Image.open(str(mf)).convert("L"), dtype=np.uint8)
            except Exception:
                pass
    except Exception:
        pass
    return _FULL_MASKS_CACHE


def _get_torch_visual_hull(device):
    """Per-view tensors (R, t, fx, fy, cx, cy, mask) on GPU for fast Gaussian projection."""
    global _TORCH_VH_CACHE
    if _TORCH_VH_CACHE is not None:
        return _TORCH_VH_CACHE
    _TORCH_VH_CACHE = False
    ctx = _get_visual_hull_context()
    if not ctx:
        return False
    try:
        import numpy as np
        import torch

        views, mask_by_stem = ctx
        packed = []
        for img_name, R, t, (cw, ch, fx, fy, cx, cy) in views:
            m = mask_by_stem.get(img_name)
            if m is None:
                m = mask_by_stem.get(Path(img_name).stem)
            if m is None:
                continue
            mh, mw = m.shape[:2]
            sx, sy = float(mw) / max(1, cw), float(mh) / max(1, ch)
            packed.append(
                (
                    torch.tensor(R, dtype=torch.float32, device=device),
                    torch.tensor(t, dtype=torch.float32, device=device),
                    float(fx * sx),
                    float(fy * sy),
                    float(cx * sx),
                    float(cy * sy),
                    torch.from_numpy((m > 127).astype(np.uint8)).to(device),
                )
            )
        if packed:
            _TORCH_VH_CACHE = packed
    except Exception:
        _TORCH_VH_CACHE = False
    return _TORCH_VH_CACHE


def _visual_hull_outside_mask_torch(xyz):
    """Boolean tensor: True for Gaussians that lie outside the object's 3D Visual Hull."""
    import torch

    packed = _get_torch_visual_hull(xyz.device)
    if not packed:
        return None
    with torch.no_grad():
        n = xyz.shape[0]
        vis_cnt = torch.zeros(n, dtype=torch.int32, device=xyz.device)
        fg_cnt = torch.zeros(n, dtype=torch.int32, device=xyz.device)
        for R, t, fx, fy, cx, cy, m in packed:
            p = xyz @ R.T + t
            z = p[:, 2]
            valid = z > 0.05
            zs = torch.where(valid, z, torch.ones_like(z))
            u = torch.round(fx * p[:, 0] / zs + cx).long()
            v = torch.round(fy * p[:, 1] / zs + cy).long()
            mh, mw = m.shape
            inb = valid & (u >= 0) & (u < mw) & (v >= 0) & (v < mh)
            uc = u.clamp(0, mw - 1)
            vc = v.clamp(0, mh - 1)
            fg = (m[vc, uc] > 0) & inb
            vis_cnt += inb.int()
            fg_cnt += fg.int()
        ratio = fg_cnt.float() / vis_cnt.clamp(min=1).float()
        # A point seen by >= 3 cameras but inside the silhouette in < 50% of them is background
        return (vis_cnt >= 3) & (ratio < 0.5)


def _patch_gaussian_model_pruning(mod):
    target_cls = getattr(mod, "GaussianModel", None)
    if target_cls is None or getattr(target_cls, "_bakalarka_vh_prune_patched", False):
        return
    orig_dp = getattr(target_cls, "densify_and_prune", None)
    if not callable(orig_dp) or not callable(getattr(target_cls, "prune_points", None)):
        return
    target_cls._bakalarka_vh_prune_patched = True

    def patched_densify_and_prune(self, *a, **kw):
        res = orig_dp(self, *a, **kw)
        try:
            outside = _visual_hull_outside_mask_torch(self.get_xyz.detach())
            if outside is not None:
                n_out = int(outside.sum().item())
                n_all = int(outside.shape[0])
                # Safety: never wipe the model if masks/poses are inconsistent
                if 0 < n_out < int(n_all * 0.7):
                    self.prune_points(outside)
                    print(
                        "[VisualHull] Видалено %d / %d гаусіанів поза об'єктом під час навчання." % (n_out, n_all),
                        flush=True,
                    )
        except Exception as exc:
            print("[VisualHull] Попередження під час прунінгу: %s" % exc, flush=True)
        return res

    target_cls.densify_and_prune = patched_densify_and_prune


def _patch_mesh_extractor(mod):
    """Zero rendered depth outside the (slightly dilated) foreground mask before TSDF fusion."""
    target_cls = getattr(mod, "GaussianExtractor", None)
    if target_cls is None or getattr(target_cls, "_bakalarka_vh_extract_patched", False):
        return
    orig_recon = getattr(target_cls, "reconstruction", None)
    if not callable(orig_recon):
        return
    target_cls._bakalarka_vh_extract_patched = True

    def patched_reconstruction(self, *a, **kw):
        res = orig_recon(self, *a, **kw)
        masks = _get_all_masks_by_stem()
        if not masks:
            return res
        try:
            import cv2
            import numpy as np
            import torch

            cams = getattr(self, "viewpoint_stack", None) or []
            depthmaps = getattr(self, "depthmaps", None)
            alphamaps = getattr(self, "alphamaps", None)
            if not depthmaps:
                return res
            masked = 0
            for i, cam in enumerate(cams):
                if i >= len(depthmaps):
                    break
                stem = Path(str(getattr(cam, "image_name", ""))).stem
                m = masks.get(stem)
                if m is None:
                    continue
                d = depthmaps[i]
                h, w = int(d.shape[-2]), int(d.shape[-1])
                mr = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                k = max(3, int(round(min(h, w) * 0.01)) * 2 + 1)
                mr = cv2.dilate(mr, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
                keep = torch.from_numpy((mr > 127).astype(np.float32)).to(d.device)
                depthmaps[i] = d * keep.view(*([1] * (d.dim() - 2)), h, w)
                if alphamaps is not None and i < len(alphamaps):
                    a_t = alphamaps[i]
                    alphamaps[i] = a_t * keep.to(a_t.device).view(*([1] * (a_t.dim() - 2)), h, w)
                masked += 1
            print("[VisualHull] Замасковано фон у %d картах глибини перед TSDF-злиттям." % masked, flush=True)
        except Exception as exc:
            print("[VisualHull] Попередження при маскуванні глибини: %s" % exc, flush=True)
        return res

    target_cls.reconstruction = patched_reconstruction


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

        if _watertight_flag and mesh_path and os.path.isfile(str(mesh_path)):
            make_mesh_watertight(mesh_path)

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

        if _watertight_flag and output_dir and os.path.isdir(str(output_dir)):
            for ext in ("*.ply", "*.obj"):
                for f in sorted(Path(output_dir).rglob(ext)):
                    if not f.name.startswith("point_cloud"):
                        make_mesh_watertight(f)

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

        # Hook 2: open3d.io.write_triangle_mesh for immediate in-memory watertight repair
        if _watertight_flag and (name == "open3d" or name.startswith("open3d.")):
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

        # Hook 5: Visual-Hull Gaussian pruning during training + depth masking before TSDF mesh fusion
        if _mask_fg_flag:
            try:
                if "gaussian_model" in name or name == "scene" or name.endswith(".scene"):
                    for cand in (mod, sys.modules.get("scene.gaussian_model"), getattr(mod, "gaussian_model", None)):
                        if cand is not None:
                            _patch_gaussian_model_pruning(cand)
                if "mesh_utils" in name or name == "utils" or name.endswith(".utils"):
                    for cand in (mod, sys.modules.get("utils.mesh_utils"), getattr(mod, "mesh_utils", None)):
                        if cand is not None:
                            _patch_mesh_extractor(cand)
            except Exception:
                pass

        # Hook 4: scene.point_generator / scene.dataset_readers for Visual-Hull point filtering
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
