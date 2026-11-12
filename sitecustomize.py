"""
Automatic hook for 3D Gaussian Splatting / 2D-SuGaR training:
1. Non-blocking LiveStreamer telemetry & point-cloud streaming (when LIVE_STREAM_PACKET is set)
2. Configurable SuGaR Mesh Refinement mode (REFINEMENT_MODE = skip | fast | balanced | full)
3. Automatic Watertight Mesh hole-filling & manifold repair (when WATERTIGHT_MESH=1)
Loaded automatically via PYTHONPATH by Python site module.
"""
import builtins
import os
from pathlib import Path
import shutil
import sys


def _close_mesh_boundaries_numpy(verts, faces, colors=None):
    """
    Pure NumPy topological boundary-loop detector and fan triangulator.
    Closes all open boundary loops (holes of any size, including object bases)
    and interpolates vertex colors for newly added cap centroids.
    Returns (repaired_verts, repaired_faces, repaired_colors, holes_closed_count).
    """
    import numpy as np
    from collections import Counter, defaultdict

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(verts) == 0 or len(faces) == 0:
        return verts, faces, colors, 0

    # Remove degenerate triangles (where two or more vertex indices are identical)
    valid_mask = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 2] != faces[:, 0])
    )
    faces = faces[valid_mask]
    if len(faces) == 0:
        return verts, faces, colors, 0

    # Remove exact duplicate triangles (ignoring winding / cyclic shift)
    _, unique_idx = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(unique_idx)]

    verts_list = [verts]
    colors_list = [colors] if (colors is not None and len(colors) == len(verts)) else None
    total_loops_closed = 0

    # Run up to 3 passes to guarantee all simple & pinched boundary cycles are sealed
    for _pass in range(3):
        h0 = faces[:, [0, 1]]
        h1 = faces[:, [1, 2]]
        h2 = faces[:, [2, 0]]
        all_he = np.vstack([h0, h1, h2])
        he_counts = Counter(map(tuple, all_he.tolist()))

        adj = defaultdict(list)
        open_edge_count = 0
        for (u, v), cnt in he_counts.items():
            rev_cnt = he_counts.get((v, u), 0)
            if cnt > rev_cnt:
                for _ in range(cnt - rev_cnt):
                    # Missing opposite half-edge is (v -> u)
                    adj[int(v)].append(int(u))
                    open_edge_count += 1

        if open_edge_count == 0:
            break

        cur_verts = np.vstack(verts_list)
        cur_colors = np.vstack(colors_list) if colors_list is not None else None
        next_vid = int(cur_verts.shape[0])
        new_faces = [faces]
        loops_in_pass = 0

        while adj:
            start_node = next(iter(adj))
            path_nodes = [start_node]
            visited_idx = {start_node: 0}
            curr = start_node

            while curr is not None:
                if curr not in adj or not adj[curr]:
                    # Dead-end on non-manifold chain: close remaining chain if >= 3 or backtrack
                    if len(path_nodes) >= 3:
                        loop = path_nodes[:]
                        if len(loop) == 3:
                            new_faces.append(np.array([[loop[0], loop[1], loop[2]]], dtype=np.int64))
                        else:
                            idx_arr = np.array(loop, dtype=np.int64)
                            centroid = cur_verts[idx_arr].mean(axis=0, keepdims=True)
                            verts_list.append(centroid)
                            if colors_list is not None and cur_colors is not None:
                                c_mean = cur_colors[idx_arr].mean(axis=0, keepdims=True)
                                if np.issubdtype(cur_colors.dtype, np.integer):
                                    c_mean = np.clip(np.round(c_mean), 0, 255).astype(cur_colors.dtype)
                                colors_list.append(c_mean)
                            center_vid = next_vid
                            next_vid += 1
                            L = len(loop)
                            fan = np.array(
                                [[loop[i], loop[(i + 1) % L], center_vid] for i in range(L)],
                                dtype=np.int64,
                            )
                            new_faces.append(fan)
                        loops_in_pass += 1
                    break

                nxt = adj[curr].pop()
                if not adj[curr]:
                    del adj[curr]

                if nxt in visited_idx:
                    loop_start = visited_idx[nxt]
                    loop = path_nodes[loop_start:]
                    if len(loop) == 3:
                        new_faces.append(np.array([[loop[0], loop[1], loop[2]]], dtype=np.int64))
                        loops_in_pass += 1
                    elif len(loop) >= 4:
                        idx_arr = np.array(loop, dtype=np.int64)
                        centroid = cur_verts[idx_arr].mean(axis=0, keepdims=True)
                        verts_list.append(centroid)
                        if colors_list is not None and cur_colors is not None:
                            c_mean = cur_colors[idx_arr].mean(axis=0, keepdims=True)
                            if np.issubdtype(cur_colors.dtype, np.integer):
                                c_mean = np.clip(np.round(c_mean), 0, 255).astype(cur_colors.dtype)
                            colors_list.append(c_mean)
                        center_vid = next_vid
                        next_vid += 1
                        L = len(loop)
                        fan = np.array(
                            [[loop[i], loop[(i + 1) % L], center_vid] for i in range(L)],
                            dtype=np.int64,
                        )
                        new_faces.append(fan)
                        loops_in_pass += 1

                    for removed in path_nodes[loop_start + 1 :]:
                        visited_idx.pop(removed, None)
                    path_nodes = path_nodes[: loop_start + 1]
                    curr = nxt
                else:
                    visited_idx[nxt] = len(path_nodes)
                    path_nodes.append(nxt)
                    curr = nxt

        faces = np.vstack(new_faces)
        total_loops_closed += loops_in_pass
        if loops_in_pass == 0:
            break

    final_verts = np.vstack(verts_list)
    final_colors = np.vstack(colors_list) if colors_list is not None else None
    return final_verts, faces, final_colors, total_loops_closed


def _make_obj_watertight(path: Path) -> bool:
    """
    Close all open boundary loops in a Wavefront .obj file while preserving
    existing UV coordinates (vt), normals (vn), mtllib, and usemtl directives.
    """
    import numpy as np

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
                # Triangulate polygon face if > 3 vertices
                for i in range(1, len(fv) - 1):
                    faces_v.append([fv[0], fv[i], fv[i + 1]])

    if not verts or not faces_v:
        return False

    orig_v_count = len(verts)
    orig_f_count = len(faces_v)
    verts_arr = np.asarray(verts, dtype=np.float64)
    faces_arr = np.asarray(faces_v, dtype=np.int64)

    repaired_verts, repaired_faces, _, loops_closed = _close_mesh_boundaries_numpy(
        verts_arr, faces_arr, None
    )
    if loops_closed == 0 and len(repaired_faces) == orig_f_count:
        print(f"[Watertight] OBJ-сітка {path.name} вже герметична.", flush=True)
        return True

    added_verts = repaired_verts[orig_v_count:]
    added_faces = repaired_faces[orig_f_count:]

    append_lines = ["", "# Watertight hole-closing cap vertices and faces"]
    for v in added_verts:
        append_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    has_vt = len(vts) > 0 and len(face_v_to_vt) > 0
    vts_arr = np.asarray(vts, dtype=np.float64) if has_vt else None
    next_vt_idx = len(vts) + 1
    center_vid_to_vt_idx = {}

    if has_vt and len(added_verts) > 0:
        # Estimate UV for each new centroid vertex from its adjacent loop vertices
        for f in added_faces:
            for vi in f:
                if vi >= orig_v_count and vi not in center_vid_to_vt_idx:
                    neighbor_vts = [
                        vts_arr[face_v_to_vt[n]]
                        for n in f
                        if n < orig_v_count and n in face_v_to_vt and face_v_to_vt[n] < len(vts_arr)
                    ]
                    if neighbor_vts:
                        uv = np.mean(neighbor_vts, axis=0)
                    else:
                        uv = np.array([0.5, 0.5])
                    append_lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")
                    center_vid_to_vt_idx[vi] = next_vt_idx
                    next_vt_idx += 1

    for f in added_faces:
        v1, v2, v3 = int(f[0]) + 1, int(f[1]) + 1, int(f[2]) + 1
        if has_vt:
            t1 = (face_v_to_vt.get(int(f[0]), 0) + 1) if int(f[0]) < orig_v_count else center_vid_to_vt_idx.get(int(f[0]), 1)
            t2 = (face_v_to_vt.get(int(f[1]), 0) + 1) if int(f[1]) < orig_v_count else center_vid_to_vt_idx.get(int(f[1]), 1)
            t3 = (face_v_to_vt.get(int(f[2]), 0) + 1) if int(f[2]) < orig_v_count else center_vid_to_vt_idx.get(int(f[2]), 1)
            append_lines.append(f"f {v1}/{t1} {v2}/{t2} {v3}/{t3}")
        else:
            append_lines.append(f"f {v1} {v2} {v3}")

    with open(path, "a", encoding="utf-8") as f_out:
        f_out.write("\n".join(append_lines) + "\n")

    print(
        f"[Watertight] Зашито {loops_closed} контурів у {path.name}: +{len(added_verts)} V, +{len(added_faces)} F.",
        flush=True,
    )
    return True


_IN_WATERTIGHT_WRITE = False


def make_mesh_watertight(mesh_file: str | Path) -> bool:
    """
    Close all open boundary loops (holes) in a triangle mesh (.ply or .obj)
    while preserving vertex colors (for .ply) or UV textures (for .obj).
    Uses Open3D + NumPy (installed natively in 2d-sugar environment) with a
    plyfile fallback, requiring zero external dependencies like trimesh.
    """
    global _IN_WATERTIGHT_WRITE
    if _IN_WATERTIGHT_WRITE:
        return False

    path = Path(mesh_file)
    if not path.is_file():
        return False

    suffix = path.suffix.lower()
    if suffix == ".obj":
        try:
            return _make_obj_watertight(path)
        except Exception as exc:
            print(f"[Watertight] Попередження під час обробки OBJ {path.name}: {exc}", flush=True)
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
            faces = np.asarray(mesh.triangles, dtype=np.int64)
            if len(faces) == 0:
                return False

            verts = np.asarray(mesh.vertices, dtype=np.float64)
            has_colors = mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(verts)
            colors = np.asarray(mesh.vertex_colors, dtype=np.float64) if has_colors else None

            print(
                f"[Watertight] Перевірка та герметизація сітки: {path.name} ({len(verts)} V, {len(faces)} F)...",
                flush=True,
            )

            repaired_verts, repaired_faces, repaired_colors, loops_closed = _close_mesh_boundaries_numpy(
                verts, faces, colors
            )

            out_mesh = o3d.geometry.TriangleMesh()
            out_mesh.vertices = o3d.utility.Vector3dVector(repaired_verts)
            out_mesh.triangles = o3d.utility.Vector3iVector(repaired_faces.astype(np.int32))
            if repaired_colors is not None and len(repaired_colors) == len(repaired_verts):
                out_mesh.vertex_colors = o3d.utility.Vector3dVector(repaired_colors)
            out_mesh.remove_unreferenced_vertices()
            out_mesh.compute_vertex_normals()

            o3d.io.write_triangle_mesh(str(path), out_mesh)
            print(
                f"[Watertight] Готово ({path.name}): зашито {loops_closed} дірок -> {len(out_mesh.vertices)} V, {len(out_mesh.triangles)} F.",
                flush=True,
            )
            return True
        except ImportError:
            pass

        # Strategy 2: plyfile fallback (also standard in 3DGS / 2d-sugar)
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
            f"[Watertight] Готово ({path.name}, plyfile): зашито {loops_closed} дірок -> {len(repaired_verts)} V, {len(repaired_faces)} F.",
            flush=True,
        )
        return True
    except Exception as exc:
        print(f"[Watertight] Попередження під час обробки {path}: {exc}", flush=True)
        return False
    finally:
        _IN_WATERTIGHT_WRITE = False


# 1. Resolve REFINEMENT_MODE and WATERTIGHT_MESH (including fallback from composite SCENE_TYPE="object:skip:1")
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

if len(sys.argv) > 0:
    _script_name = os.path.basename(sys.argv[0])
    if "train_refined" in _script_name:
        if _refinement_mode in ("skip", "0", "none", "false"):
            print("\n========== [2D-SuGaR] REFINEMENT_MODE=skip: Пропуск train_refined.py ==========", flush=True)
            sys.exit(0)
        elif _refinement_mode in ("fast", "2000", "balanced", "7000"):
            target_iters = "2000" if _refinement_mode in ("fast", "2000") else "7000"
            if "-f" in sys.argv:
                idx = sys.argv.index("-f")
                if idx + 1 < len(sys.argv):
                    sys.argv[idx + 1] = target_iters
            elif "--refinement_iterations" in sys.argv:
                idx = sys.argv.index("--refinement_iterations")
                if idx + 1 < len(sys.argv):
                    sys.argv[idx + 1] = target_iters
            else:
                sys.argv.extend(["-f", target_iters])
    elif _script_name == "train.py":
        if _refinement_mode in ("skip", "0", "none", "false"):
            if "--skip_sugar_refine" not in sys.argv:
                sys.argv.append("--skip_sugar_refine")
        elif _refinement_mode in ("fast", "2000", "balanced", "7000"):
            target_iters = "2000" if _refinement_mode in ("fast", "2000") else "7000"
            if "-f" not in sys.argv and "--refinement_iterations" not in sys.argv:
                sys.argv.extend(["-f", target_iters])


# 2. Initialize LiveStreamer if LIVE_STREAM_PACKET is set
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
        res = orig_write(filename, mesh, *w_args, **w_kwargs)
        if not _IN_WATERTIGHT_WRITE and str(filename).lower().endswith("fuse_post.ply"):
            try:
                make_mesh_watertight(str(filename))
            except Exception:
                pass
        return res

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

        # Locate fuse_post.ply if mesh_path not explicitly found
        if not mesh_path or not os.path.isfile(str(mesh_path)):
            model_dir = os.getenv("MODEL_OUTPUT_DIR")
            if model_dir and os.path.isdir(model_dir):
                candidates = sorted(Path(model_dir).rglob("fuse_post.ply")) or sorted(Path(model_dir).rglob("fuse.ply"))
                if candidates:
                    mesh_path = str(candidates[-1])

        # Apply watertight repair on fuse_post.ply right after 2DGS+TSDF (before refinement)
        if _watertight_flag and mesh_path and os.path.isfile(str(mesh_path)):
            make_mesh_watertight(mesh_path)

        # Check REFINEMENT_MODE
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
                print(f"[2D-SuGaR] Готову TSDF-сітку збережено у: {dst_ply}", flush=True)
            print("Iteration: 15000\nloss: 0.000000  [ 15000/15000] computed in 0.01 minutes.", flush=True)
            return None

        if _refinement_mode in ("fast", "2000", "balanced", "7000"):
            target_iters = 2000 if _refinement_mode in ("fast", "2000") else 7000
            print(
                f"\n========== [2D-SuGaR] REFINEMENT_MODE={_refinement_mode}: Запуск SuGaR Refinement на {target_iters} ітерацій ==========",
                flush=True,
            )
            if cfg is not None:
                setattr(cfg, "refinement_iterations", target_iters)
            if "refinement_iterations" in r_kwargs:
                r_kwargs["refinement_iterations"] = target_iters

        res = orig_refined_training(*r_args, **r_kwargs)

        # If watertight is enabled, also repair final meshes exported to output_dir
        if _watertight_flag and output_dir and os.path.isdir(str(output_dir)):
            for ext in ("*.ply", "*.obj"):
                for f in sorted(Path(output_dir).rglob(ext)):
                    if not f.name.startswith("point_cloud"):
                        make_mesh_watertight(f)

        return res

    refine_mod.refined_training = patched_refined_training


# 3. Install import hook for GaussianModel (LiveStreamer), open3d, and sugar_trainers.refine
try:
    _orig_import = builtins.__import__

    def _custom_import(name, *args, **kwargs):
        mod = _orig_import(name, *args, **kwargs)

        # Hook 1: GaussianModel optimizer step for LiveStreamer
        if streamer is not None and (name == "scene.gaussian_model" or name == "scene"):
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

        # Hook 2: open3d.io.write_triangle_mesh for immediate fuse_post.ply watertight repair
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
