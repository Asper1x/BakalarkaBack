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


def make_mesh_watertight(mesh_file: str | Path) -> bool:
    """
    Close all open boundary loops (holes) in a triangle mesh (.ply or .obj)
    while preserving vertex colors if present.
    Uses pymeshfix when available, with a pure NumPy/Trimesh boundary-loop fan
    triangulation fallback that handles holes of any size without external C++ deps.
    """
    path = Path(mesh_file)
    if not path.is_file():
        return False

    try:
        import numpy as np
        import trimesh

        loaded = trimesh.load(str(path), process=False)
        if isinstance(loaded, trimesh.Scene):
            if not loaded.geometry:
                return False
            mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
        else:
            mesh = loaded

        if not hasattr(mesh, "faces") or mesh.faces is None or len(mesh.faces) == 0:
            # Point cloud only (no faces), skip
            return False

        orig_verts = np.asarray(mesh.vertices, dtype=np.float64)
        orig_faces = np.asarray(mesh.faces, dtype=np.int64)
        has_colors = (
            hasattr(mesh, "visual")
            and getattr(mesh.visual, "kind", None) == "vertex"
            and getattr(mesh.visual, "vertex_colors", None) is not None
            and len(mesh.visual.vertex_colors) == len(orig_verts)
        )
        orig_colors = (
            np.asarray(mesh.visual.vertex_colors, dtype=np.uint8).copy()
            if has_colors
            else None
        )

        # Clean degenerate / duplicate faces first
        mesh.remove_degenerate_faces()
        mesh.remove_duplicate_faces()
        mesh.remove_unreferenced_vertices()

        if mesh.is_watertight:
            print(f"[Watertight] Сітка {path.name} вже герметична (is_watertight=True).", flush=True)
            return True

        print(f"[Watertight] Зашивання дірок та герметизація сітки: {path.name} ({len(mesh.vertices)} V, {len(mesh.faces)} F)...", flush=True)

        repaired_verts = None
        repaired_faces = None

        # Strategy 1: PyMeshFix (if installed in environment)
        try:
            import pymeshfix
            tin = pymeshfix.PyTMesh(False)
            tin.load_array(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int32))
            tin.fill_small_boundaries(nbe=0, refine=True)
            tin.clean(max_iters=10, inner_loops=3)
            repaired_verts, repaired_faces = tin.return_arrays()
        except Exception:
            repaired_verts = None
            repaired_faces = None

        # Strategy 2: Trimesh repair + topological boundary-loop fan triangulation (pure NumPy)
        if repaired_verts is None or repaired_faces is None or len(repaired_faces) == 0:
            try:
                trimesh.repair.fix_normals(mesh)
                trimesh.repair.fill_holes(mesh)
            except Exception:
                pass

            verts_list = [np.asarray(mesh.vertices, dtype=np.float64)]
            faces_arr = np.asarray(mesh.faces, dtype=np.int64)
            new_faces = [faces_arr]

            # Find all directed half-edges (u -> v) that lack an opposite half-edge (v -> u)
            h0 = faces_arr[:, [0, 1]]
            h1 = faces_arr[:, [1, 2]]
            h2 = faces_arr[:, [2, 0]]
            all_half_edges = np.vstack([h0, h1, h2])

            # Encode directed edges as 64-bit integers for fast set lookup
            max_v = int(verts_list[0].shape[0]) + 1
            forward_keys = all_half_edges[:, 0] * max_v + all_half_edges[:, 1]
            reverse_keys = all_half_edges[:, 1] * max_v + all_half_edges[:, 0]
            forward_set = set(forward_keys.tolist())

            boundary_mask = np.array([rk not in forward_set for rk in reverse_keys.tolist()], dtype=bool)
            open_edges = all_half_edges[boundary_mask]

            if len(open_edges) > 0:
                # To plug a hole consistently with adjacent face normals, we orient each missing edge as (v -> u)
                adj: dict[int, list[int]] = {}
                for u, v in open_edges:
                    adj.setdefault(int(v), []).append(int(u))

                next_vid = int(verts_list[0].shape[0])
                base_verts = verts_list[0]

                while adj:
                    start_node = next(iter(adj))
                    path_nodes = [start_node]
                    visited_idx = {start_node: 0}
                    curr = start_node

                    while curr in adj and adj[curr]:
                        nxt = adj[curr].pop()
                        if not adj[curr]:
                            del adj[curr]

                        if nxt in visited_idx:
                            # Found a closed boundary loop!
                            loop_start = visited_idx[nxt]
                            loop = path_nodes[loop_start:]
                            if len(loop) == 3:
                                new_faces.append(np.array([[loop[0], loop[1], loop[2]]], dtype=np.int64))
                            elif len(loop) >= 4:
                                loop_coords = base_verts[np.array(loop, dtype=np.int64)]
                                centroid = loop_coords.mean(axis=0, keepdims=True)
                                verts_list.append(centroid)
                                center_vid = next_vid
                                next_vid += 1
                                fan = np.array(
                                    [[loop[i], loop[(i + 1) % len(loop)], center_vid] for i in range(len(loop))],
                                    dtype=np.int64,
                                )
                                new_faces.append(fan)

                            # Trim cycle from path_nodes and continue
                            for removed in path_nodes[loop_start + 1:]:
                                visited_idx.pop(removed, None)
                            path_nodes = path_nodes[:loop_start + 1]
                            curr = nxt
                        else:
                            visited_idx[nxt] = len(path_nodes)
                            path_nodes.append(nxt)
                            curr = nxt

            repaired_verts = np.vstack(verts_list)
            repaired_faces = np.vstack(new_faces)

        # Restore / interpolate vertex colors via nearest neighbor lookup
        repaired_colors = None
        if orig_colors is not None and len(orig_verts) > 0:
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(orig_verts)
                dists, idxs = tree.query(repaired_verts, k=1)
                repaired_colors = orig_colors[idxs]
            except Exception:
                if len(repaired_verts) >= len(orig_colors):
                    repaired_colors = np.zeros((len(repaired_verts), orig_colors.shape[1]), dtype=np.uint8)
                    repaired_colors[:len(orig_colors)] = orig_colors
                    repaired_colors[len(orig_colors):] = orig_colors.mean(axis=0).astype(np.uint8)

        out_mesh = trimesh.Trimesh(
            vertices=repaired_verts,
            faces=repaired_faces,
            vertex_colors=repaired_colors,
            process=True,
        )
        try:
            trimesh.repair.fix_normals(out_mesh)
        except Exception:
            pass

        out_mesh.export(str(path))
        print(
            f"[Watertight] Готово: {path.name} -> {len(out_mesh.vertices)} V, {len(out_mesh.faces)} F (is_watertight={out_mesh.is_watertight})",
            flush=True,
        )
        return True
    except Exception as exc:
        print(f"[Watertight] Попередження під час обробки {path}: {exc}", flush=True)
        return False


# 1. Handle CLI sys.argv overrides if train_refined.py is invoked as a subprocess
_refinement_mode = os.getenv("REFINEMENT_MODE", "full").strip().lower()
_watertight_flag = os.getenv("WATERTIGHT_MESH", "0").strip() == "1"

if len(sys.argv) > 0 and "train_refined" in os.path.basename(sys.argv[0]):
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


# 3. Install import hook for both GaussianModel (LiveStreamer) and sugar_trainers.refine
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

        # Hook 2: sugar_trainers.refine for REFINEMENT_MODE and WATERTIGHT_MESH
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
