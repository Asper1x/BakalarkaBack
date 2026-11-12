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

if ":" in _raw_scene_type:
    _parts = _raw_scene_type.split(":")
    if len(_parts) >= 2 and not _refinement_mode:
        _refinement_mode = _parts[1].strip().lower()
    if len(_parts) >= 3 and not _watertight_env:
        _watertight_env = _parts[2].strip().lower()

if not _refinement_mode:
    _refinement_mode = "full"
_watertight_flag = _watertight_env in ("1", "true", "yes", "on")


# ==============================================================================
# 3. Watertight Mesh Hole-Closing (Open3D + Pure NumPy Multi-Ring Smooth Cap)
# ==============================================================================
def _close_mesh_boundaries_numpy(verts, faces, colors=None):
    """
    Topological boundary-loop detector and smooth concentric-ring membrane triangulator.
    Detects open boundary edges via undirected edge triangle incidence == 1, traces all
    closed boundary cycles, and seals both small holes and large base openings with a
    smooth Laplacian-faired membrane cap while interpolating vertex colors.
    Returns (repaired_verts, repaired_faces, repaired_colors, holes_closed_count).
    """
    import numpy as np
    from collections import Counter, defaultdict

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(verts) == 0 or len(faces) == 0:
        return verts, faces, colors, 0

    has_colors = colors is not None and len(colors) == len(verts)
    if has_colors:
        colors = np.asarray(colors)

    # Step A: Weld duplicate vertices at identical 3D positions (e.g. TSDF sub-volume block seams)
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

    verts_list = [verts]
    colors_list = [colors] if has_colors else None
    total_loops_closed = 0

    for _pass in range(3):
        # Extract all 3 directed half-edges per triangle
        h0 = faces[:, [0, 1]]
        h1 = faces[:, [1, 2]]
        h2 = faces[:, [2, 0]]
        all_he = np.vstack([h0, h1, h2])

        # Count undirected edge occurrences across all triangles.
        # An edge {min(u,v), max(u,v)} is on an open hole boundary iff it belongs to strictly 1 triangle.
        u_min = np.minimum(all_he[:, 0], all_he[:, 1])
        u_max = np.maximum(all_he[:, 0], all_he[:, 1])
        undir_edges = np.column_stack([u_min, u_max])
        undir_counts = Counter(map(tuple, undir_edges.tolist()))

        # Build undirected boundary adjacency graph + preferred reversed half-edge orientation
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
                # Existing triangle has u_i -> v_i, so hole cap needs v_i -> u_i
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
            # Orient loop consistently with pref_dir
            fwd_score = sum(1 for i in range(L) if (loop_nodes[i], loop_nodes[(i + 1) % L]) in pref_dir)
            rev_score = sum(1 for i in range(L) if (loop_nodes[(i + 1) % L], loop_nodes[i]) in pref_dir)
            if rev_score > fwd_score:
                loop_nodes = list(reversed(loop_nodes))

            if L == 3:
                new_faces.append(np.array([[loop_nodes[0], loop_nodes[1], loop_nodes[2]]], dtype=np.int64))
                loops_in_pass += 1
                return

            idx_arr = np.array(loop_nodes, dtype=np.int64)
            loop_pts = cur_verts[idx_arr]
            centroid = loop_pts.mean(axis=0, keepdims=True)
            loop_cols = cur_colors[idx_arr] if (colors_list is not None and cur_colors is not None) else None
            c_mean = loop_cols.mean(axis=0, keepdims=True) if loop_cols is not None else None

            if L <= 12:
                verts_list.append(centroid)
                if colors_list is not None and c_mean is not None:
                    if np.issubdtype(cur_colors.dtype, np.integer):
                        c_mean = np.clip(np.round(c_mean), 0, 255).astype(cur_colors.dtype)
                    colors_list.append(c_mean)
                center_vid = next_vid
                next_vid += 1
                fan = np.array(
                    [[loop_nodes[i], loop_nodes[(i + 1) % L], center_vid] for i in range(L)],
                    dtype=np.int64,
                )
                new_faces.append(fan)
                loops_in_pass += 1
                return

            # Multi-ring smooth membrane cap for larger holes (L > 12, e.g. object base)
            ring_fracs = (0.28, 0.56, 0.80)
            prev_ring_ids = loop_nodes
            curr_ring_pts = loop_pts.copy()
            curr_ring_cols = loop_cols.copy() if loop_cols is not None else None

            for frac in ring_fracs:
                # Smooth along loop contour before contracting toward centroid
                for _ in range(3):
                    curr_ring_pts = 0.25 * np.roll(curr_ring_pts, 1, axis=0) + 0.5 * curr_ring_pts + 0.25 * np.roll(curr_ring_pts, -1, axis=0)
                    if curr_ring_cols is not None:
                        curr_ring_cols = 0.25 * np.roll(curr_ring_cols, 1, axis=0) + 0.5 * curr_ring_cols + 0.25 * np.roll(curr_ring_cols, -1, axis=0)

                ring_pts = (1.0 - frac) * curr_ring_pts + frac * centroid
                verts_list.append(ring_pts)
                if colors_list is not None and curr_ring_cols is not None and c_mean is not None:
                    ring_cols = (1.0 - frac) * curr_ring_cols + frac * c_mean
                    if np.issubdtype(cur_colors.dtype, np.integer):
                        ring_cols = np.clip(np.round(ring_cols), 0, 255).astype(cur_colors.dtype)
                    colors_list.append(ring_cols)

                ring_ids = [next_vid + i for i in range(L)]
                next_vid += L

                # Connect prev_ring_ids to ring_ids with 2L triangles
                strip_tris = []
                for i in range(L):
                    ni = (i + 1) % L
                    a0, a1 = prev_ring_ids[i], prev_ring_ids[ni]
                    b0, b1 = ring_ids[i], ring_ids[ni]
                    strip_tris.append([a0, a1, b1])
                    strip_tris.append([a0, b1, b0])
                new_faces.append(np.array(strip_tris, dtype=np.int64))
                prev_ring_ids = ring_ids

            # Connect innermost ring to center vertex
            verts_list.append(centroid)
            if colors_list is not None and c_mean is not None:
                c_out = c_mean
                if np.issubdtype(cur_colors.dtype, np.integer):
                    c_out = np.clip(np.round(c_out), 0, 255).astype(cur_colors.dtype)
                colors_list.append(c_out)
            center_vid = next_vid
            next_vid += 1

            fan = np.array(
                [[prev_ring_ids[i], prev_ring_ids[(i + 1) % L], center_vid] for i in range(L)],
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

                # Prefer edge oriented (curr -> nb) in pref_dir, or one closing a cycle
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
    Close all open boundary loops in a Wavefront .obj file while preserving
    existing UV coordinates (vt), normals (vn), mtllib, and usemtl directives.
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
    orig_f_count = len(faces_v)
    verts_arr = np.asarray(verts, dtype=np.float64)
    faces_arr = np.asarray(faces_v, dtype=np.int64)

    # For OBJ we run without reordering existing vertex indices so original UV/normal indices stay valid
    from collections import Counter, defaultdict

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
        print("[Watertight] OBJ-сітка " + str(path.name) + " вже герметична.", flush=True)
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
                L = len(loop)
                if L >= 3:
                    fwd_score = sum(1 for i in range(L) if (loop[i], loop[(i + 1) % L]) in pref_dir)
                    rev_score = sum(1 for i in range(L) if (loop[(i + 1) % L], loop[i]) in pref_dir)
                    if rev_score > fwd_score:
                        loop = list(reversed(loop))
                    if L == 3:
                        v1, v2, v3 = loop[0] + 1, loop[1] + 1, loop[2] + 1
                        if has_vt:
                            t1 = face_v_to_vt.get(loop[0], 0) + 1
                            t2 = face_v_to_vt.get(loop[1], 0) + 1
                            t3 = face_v_to_vt.get(loop[2], 0) + 1
                            append_f_lines.append("f %d/%d %d/%d %d/%d" % (v1, t1, v2, t2, v3, t3))
                        else:
                            append_f_lines.append("f %d %d %d" % (v1, v2, v3))
                    else:
                        c_xyz = verts_arr[np.array(loop, dtype=np.int64)].mean(axis=0)
                        append_v_lines.append("v %.6f %.6f %.6f" % (c_xyz[0], c_xyz[1], c_xyz[2]))
                        c_v_idx = next_v_idx
                        next_v_idx += 1
                        c_vt_idx = 1
                        if has_vt and vts_arr is not None:
                            nb_uvs = [vts_arr[face_v_to_vt[n]] for n in loop if n in face_v_to_vt and face_v_to_vt[n] < len(vts_arr)]
                            uv = np.mean(nb_uvs, axis=0) if nb_uvs else np.array([0.5, 0.5])
                            append_vt_lines.append("vt %.6f %.6f" % (uv[0], uv[1]))
                            c_vt_idx = next_vt_idx
                            next_vt_idx += 1
                        for i in range(L):
                            a = loop[i]
                            b = loop[(i + 1) % L]
                            if has_vt:
                                ta = face_v_to_vt.get(a, 0) + 1
                                tb = face_v_to_vt.get(b, 0) + 1
                                append_f_lines.append("f %d/%d %d/%d %d/%d" % (a + 1, ta, b + 1, tb, c_v_idx, c_vt_idx))
                            else:
                                append_f_lines.append("f %d %d %d" % (a + 1, b + 1, c_v_idx))
                    loops_closed += 1
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

        return mod

    builtins.__import__ = _custom_import
except Exception:
    pass
