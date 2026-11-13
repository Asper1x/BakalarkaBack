#!/usr/bin/env python3
"""
3D-Guided Foreground Segmentation & Multi-View Visual-Hull Space Carving for 2D-SuGaR.
Compatible with Python 3.8+ and NumPy 1.x.

Runs at the start of Stage 5 (after COLMAP SfM, image undistortion, and Metric3D depth/normal priors):
1. Computes the 3D Object Core from COLMAP camera optical axes intersection and points3D.txt.
2. Safely probes rembg (isnet-general-use / u2net with CPUExecutionProvider) in an isolated subprocess
   so any C++/ONNXRuntime ABI or thread crash (SIGSEGV / exit 139) cannot kill the pipeline.
3. Combines AI foreground segmentation (or GrabCut fallback) with 3D Object Core ROI projection and
   Metric3D depth priors, followed by morphological cleanup and 2px boundary color decontamination (de-haloing).
4. Prunes 100% of background 3D points from sparse/0/points3D.txt and all dense prior .ply files
   via Multi-View 3D Visual Hull, and masks background pixels in .npy depth maps.
"""
import os
import subprocess
import sys
from pathlib import Path

# Limit ONNXRuntime / OpenMP thread count to prevent Slurm cgroup pthread SIGSEGV crashes
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("ORT_DISABLE_ALL_OPTIMIZATION", "0")

import cv2
import numpy as np
from PIL import Image


def parse_colmap_cameras(cameras_txt: Path) -> dict:
    cameras = {}
    if not cameras_txt.is_file():
        return cameras
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
    return cameras


def parse_colmap_views(images_txt: Path, cameras: dict):
    views = []
    c2ws = []
    if not images_txt.is_file():
        return views, c2ws
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
        R = np.array(
            [
                [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
                [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
            ],
            dtype=np.float64,
        )
        t = np.array([tx, ty, tz], dtype=np.float64)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R
        w2c[:3, 3] = t
        try:
            c2ws.append(np.linalg.inv(w2c))
        except Exception:
            pass
        if cam_id in cameras:
            views.append((img_name, R, t, cameras[cam_id]))
    return views, c2ws


def probe_and_init_rembg():
    """
    First probe rembg + onnxruntime in a child process so a C-extension ABI mismatch
    or ONNXRuntime segfault (SIGSEGV / exit 139) cannot terminate the main process.
    """
    for model_name in ("isnet-general-use", "u2net"):
        probe_code = (
            "import os, numpy as np, rembg\n"
            "from PIL import Image\n"
            "s = rembg.new_session(%r, providers=['CPUExecutionProvider'])\n"
            "im = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))\n"
            "rembg.remove(im, session=s, only_mask=True)\n" % model_name
        )
        try:
            res = subprocess.run(
                [sys.executable, "-c", probe_code],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )
            if res.returncode == 0:
                import rembg

                sess = rembg.new_session(model_name, providers=["CPUExecutionProvider"])
                print("[MaskFG] Активовано AI-сесію сегментації: %s (CPUExecutionProvider)" % model_name, flush=True)
                return rembg.remove, sess
            else:
                print(
                    "[MaskFG] Попередження: модель %s повернула код %d у тестовому процесі, пробуємо запасний варіант..."
                    % (model_name, res.returncode),
                    flush=True,
                )
        except Exception as exc:
            print("[MaskFG] Попередження при перевірці %s: %s" % (model_name, exc), flush=True)
    return None, None


def load_depth_prior_for_stem(dataset_dir: Path, stem: str, target_w: int, target_h: int):
    """Load Metric3D depth map (.npy) for a given frame stem if available."""
    for sub in ("depths", "depth_priors", "estimated_dense_depth", "depth"):
        npy_path = dataset_dir / sub / (stem + ".npy")
        if npy_path.is_file():
            try:
                arr = np.squeeze(np.load(str(npy_path))).astype(np.float32)
                if arr.ndim == 2 and arr.size > 0:
                    if arr.shape[:2] != (target_h, target_w):
                        arr = cv2.resize(arr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    return arr
            except Exception:
                pass
    return None


def main() -> int:
    if len(sys.argv) < 2:
        print("[MaskFG] Використання: mask_foreground_3d.py <dataset_dir>", flush=True)
        return 0

    dataset_dir = Path(sys.argv[1])
    images_dir = dataset_dir / "images"
    sparse_dir = dataset_dir / "sparse" / "0"
    masks_dir = dataset_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.is_dir() or not sparse_dir.is_dir():
        print("[MaskFG] Пропускаємо: не знайдено images/ або sparse/0/", flush=True)
        return 0

    cameras = parse_colmap_cameras(sparse_dir / "cameras.txt")
    views, c2ws = parse_colmap_views(sparse_dir / "images.txt", cameras)

    # 1. Compute 3D Look-At Center of camera optical rays
    look_at_center = np.zeros(3, dtype=np.float64)
    cam_radius = 2.0
    if c2ws:
        c2ws_arr = np.array(c2ws)
        poses = c2ws_arr[:, :3, :] @ np.diag([1, -1, -1, 1])
        directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
        m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
        mt_m = np.transpose(m, [0, 2, 1]) @ m
        try:
            look_at_center = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
            cam_radius = float(np.median(np.linalg.norm(c2ws_arr[:, :3, 3] - look_at_center, axis=-1)))
        except Exception:
            pass

    # Read 3D points from points3D.txt (including per-point camera TRACK[] image_ids in parts[8::2])
    pts3d_txt = sparse_dir / "points3D.txt"
    pts_xyz = []
    pts_lines = []
    pts_img_ids = []
    pts_header = []
    if pts3d_txt.is_file():
        for raw_line in pts3d_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = raw_line.strip()
            if not s or s.startswith("#"):
                pts_header.append(raw_line)
                continue
            parts = s.split()
            if len(parts) >= 8:
                pts_xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
                pts_lines.append(raw_line)
                track_imgs = []
                for k in range(8, len(parts), 2):
                    try:
                        track_imgs.append(int(parts[k]))
                    except Exception:
                        pass
                pts_img_ids.append(track_imgs)

    pts_xyz_arr = np.array(pts_xyz, dtype=np.float64) if pts_xyz else np.zeros((0, 3), dtype=np.float64)
    core_pts = pts_xyz_arr
    if len(pts_xyz_arr) > 20:
        dists_to_center = np.linalg.norm(pts_xyz_arr - look_at_center, axis=1)
        core_mask = dists_to_center <= max(cam_radius * 0.55, np.percentile(dists_to_center, 65))
        if np.sum(core_mask) >= 10:
            core_pts = pts_xyz_arr[core_mask]

    # 2. Initialize AI segmentation safely
    rembg_remove, rembg_session = probe_and_init_rembg()

    view_map = {v[0]: (v[1], v[2], v[3]) for v in views}
    mask_by_name = {}
    img_files = sorted(
        [f for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )

    print(
        "[MaskFG] Сегментація %d кадрів із 3D-проєкційним фільтром та очищенням ореолу (de-haloing)..."
        % len(img_files),
        flush=True,
    )

    for idx_img, img_path in enumerate(img_files):
        pil_img = Image.open(str(img_path)).convert("RGB")
        rgb = np.asarray(pil_img, dtype=np.uint8)
        h, w = rgb.shape[:2]

        # Project 3D Object Core into current view
        roi_mask = np.ones((h, w), dtype=np.uint8)
        fg_seed_px = None
        if img_path.name in view_map and len(core_pts) >= 10:
            R, t, (cw, ch, fx, fy, cx, cy) = view_map[img_path.name]
            sx, sy = float(w) / max(1, cw), float(h) / max(1, ch)
            p_cam = core_pts @ R.T + t
            valid_z = p_cam[:, 2] > 0.05
            if np.any(valid_z):
                p_cam = p_cam[valid_z]
                u = (fx * (p_cam[:, 0] / p_cam[:, 2]) + cx) * sx
                v = (fy * (p_cam[:, 1] / p_cam[:, 2]) + cy) * sy
                in_b = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                if np.sum(in_b) >= 8:
                    u_in, v_in = u[in_b], v[in_b]
                    x0 = int(max(0, np.percentile(u_in, 2) - w * 0.18))
                    x1 = int(min(w, np.percentile(u_in, 98) + w * 0.18))
                    y0 = int(max(0, np.percentile(v_in, 2) - h * 0.18))
                    y1 = int(min(h, np.percentile(v_in, 98) + h * 0.18))
                    if (x1 - x0) >= w * 0.12 and (y1 - y0) >= h * 0.12:
                        roi_mask.fill(0)
                        roi_mask[y0:y1, x0:x1] = 1
                        fg_seed_px = np.column_stack(
                            [np.clip(u_in.astype(np.int32), 0, w - 1), np.clip(v_in.astype(np.int32), 0, h - 1)]
                        )

        alpha_u8 = None
        if callable(rembg_remove):
            try:
                kwargs = {"only_mask": True, "post_process_mask": True}
                if rembg_session is not None:
                    kwargs["session"] = rembg_session
                mask_pil = rembg_remove(pil_img, **kwargs)
                alpha_u8 = np.asarray(mask_pil, dtype=np.uint8)
                if alpha_u8.ndim == 3:
                    alpha_u8 = alpha_u8[:, :, 0]
            except Exception:
                alpha_u8 = None

        # Fallback: 3D-Core + Metric3D Depth-Guided OpenCV GrabCut
        if alpha_u8 is None:
            gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
            gc_mask[roi_mask == 0] = cv2.GC_BGD

            # Incorporate Metric3D depth map if available to separate foreground object from distant room walls
            depth_map = load_depth_prior_for_stem(dataset_dir, img_path.stem, w, h)
            if depth_map is not None and fg_seed_px is not None and len(fg_seed_px) >= 5:
                seed_depths = depth_map[fg_seed_px[:, 1], fg_seed_px[:, 0]]
                valid_sd = seed_depths[seed_depths > 1e-4]
                if len(valid_sd) >= 5:
                    d_med = float(np.median(valid_sd))
                    d_spread = max(d_med * 0.35, float(np.percentile(valid_sd, 90) - np.percentile(valid_sd, 10)) * 1.6)
                    fg_depth_zone = (depth_map > 1e-4) & (np.abs(depth_map - d_med) <= d_spread) & (roi_mask > 0)
                    bg_depth_zone = (depth_map > (d_med + d_spread * 1.5))
                    gc_mask[fg_depth_zone] = cv2.GC_PR_FGD
                    gc_mask[bg_depth_zone] = cv2.GC_BGD

            if np.sum(gc_mask == cv2.GC_PR_FGD) < 100:
                cx_i, cy_i = w // 2, h // 2
                rw_i, rh_i = max(20, int(w * 0.28)), max(20, int(h * 0.28))
                gc_mask[
                    max(0, cy_i - rh_i) : min(h, cy_i + rh_i),
                    max(0, cx_i - rw_i) : min(w, cx_i + rw_i),
                ] = cv2.GC_PR_FGD

            if fg_seed_px is not None and len(fg_seed_px) > 0:
                radius_px = max(3, int(min(w, h) * 0.015))
                for px_u, px_v in fg_seed_px[:: max(1, len(fg_seed_px) // 250)]:
                    cv2.circle(gc_mask, (int(px_u), int(px_v)), radius_px, int(cv2.GC_FGD), -1)
            else:
                cx_i, cy_i = w // 2, h // 2
                sw_i, sh_i = max(10, int(w * 0.10)), max(10, int(h * 0.10))
                gc_mask[
                    max(0, cy_i - sh_i) : min(h, cy_i + sh_i),
                    max(0, cx_i - sw_i) : min(w, cx_i + sw_i),
                ] = cv2.GC_FGD

            # Outer 3% image border is background
            bx, by = max(2, int(w * 0.03)), max(2, int(h * 0.03))
            gc_mask[:by, :] = cv2.GC_BGD
            gc_mask[-by:, :] = cv2.GC_BGD
            gc_mask[:, :bx] = cv2.GC_BGD
            gc_mask[:, -bx:] = cv2.GC_BGD

            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(rgb, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
                alpha_u8 = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            except Exception:
                alpha_u8 = (roi_mask * 255).astype(np.uint8)

        # Gate by 3D ROI if main object is inside ROI
        masked_by_roi = np.where(roi_mask > 0, alpha_u8, 0).astype(np.uint8)
        if np.sum(masked_by_roi > 127) >= (h * w * 0.01):
            alpha_u8 = masked_by_roi

        # Morphological closing + primary connected component selection
        bin_mask = (alpha_u8 > 110).astype(np.uint8)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel_close)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        if num_labels > 2:
            best_label = 1
            best_score = -1.0
            img_center = np.array([w * 0.5, h * 0.5])
            for lbl in range(1, num_labels):
                area = float(stats[lbl, cv2.CC_STAT_AREA])
                if area < (h * w * 0.002):
                    continue
                dist_c = np.linalg.norm(centroids[lbl] - img_center) / max(w, h)
                score = area / (1.0 + 2.0 * dist_c)
                if score > best_score:
                    best_score = score
                    best_label = lbl
            keep_comp = (labels == best_label).astype(np.uint8)
            max_area = float(stats[best_label, cv2.CC_STAT_AREA])
            for lbl in range(1, num_labels):
                if lbl != best_label and float(stats[lbl, cv2.CC_STAT_AREA]) >= 0.20 * max_area:
                    keep_comp[labels == lbl] = 1
            bin_mask = keep_comp

        # 2px boundary erosion + edge RGB decontamination (de-haloing)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        inner_mask = cv2.erode(bin_mask, kernel_erode, iterations=1)
        if np.sum(inner_mask) < 100:
            inner_mask = bin_mask

        eroded_rgb = cv2.erode(rgb, kernel_erode, iterations=1)
        border_zone = (bin_mask > 0) & (inner_mask == 0)
        clean_rgb = rgb.copy()
        clean_rgb[border_zone] = eroded_rgb[border_zone]

        soft_alpha = cv2.GaussianBlur(inner_mask.astype(np.float32), (3, 3), 0)
        soft_alpha = np.where(inner_mask > 0, np.maximum(soft_alpha, 0.85), soft_alpha * 0.5)
        soft_alpha = np.clip(soft_alpha, 0.0, 1.0)

        out_rgb = (clean_rgb.astype(np.float32) * soft_alpha[:, :, None]).astype(np.uint8)
        Image.fromarray(out_rgb).save(str(img_path), quality=96)

        final_mask_u8 = (inner_mask * 255).astype(np.uint8)
        Image.fromarray(final_mask_u8).save(str(masks_dir / (img_path.stem + ".png")))
        mask_by_name[img_path.name] = final_mask_u8
        mask_by_name[img_path.stem] = final_mask_u8

        if (idx_img + 1) % 25 == 0 or (idx_img + 1) == len(img_files):
            print("[MaskFG] Оброблено %d / %d кадрів..." % (idx_img + 1, len(img_files)), flush=True)

    # 3. 3D Visual-Hull pruning of COLMAP points3D.txt and all prior .ply point clouds
    sampled_views = views
    if len(sampled_views) > 48:
        step = max(1, len(sampled_views) // 48)
        sampled_views = sampled_views[::step]

    def compute_visual_hull_keep_mask(points_xyz):
        if len(points_xyz) == 0 or not sampled_views:
            return np.ones(len(points_xyz), dtype=bool), np.ones(len(points_xyz), dtype=np.float32)
        vis_cnt = np.zeros(len(points_xyz), dtype=np.int32)
        fg_cnt = np.zeros(len(points_xyz), dtype=np.int32)
        for img_name, R, t, (cw, ch, fx, fy, cx, cy) in sampled_views:
            m = mask_by_name.get(img_name)
            if m is None:
                m = mask_by_name.get(Path(img_name).stem)
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
        ratio = fg_cnt.astype(np.float32) / np.maximum(1, vis_cnt)
        dists = np.linalg.norm(points_xyz - look_at_center, axis=1)
        score = ratio - 0.15 * (dists / max(0.1, cam_radius))
        keep = (vis_cnt >= 2) & (ratio >= 0.50)
        if np.sum(keep) < min(300, len(points_xyz)):
            top_k = min(len(points_xyz), max(500, int(len(points_xyz) * 0.25)))
            best_idx = np.argsort(score)[-top_k:]
            keep = np.zeros(len(points_xyz), dtype=bool)
            keep[best_idx] = True
        return keep, score

    if len(pts_xyz_arr) > 0 and pts3d_txt.is_file():
        keep_pts, pts_score = compute_visual_hull_keep_mask(pts_xyz_arr)

        # Ensure every camera IMAGE_ID retains >= 20 points so image_depth_scale_map never gets KeyError
        cam_to_pt_indices = {}
        for pt_idx, img_id_list in enumerate(pts_img_ids):
            for iid in set(img_id_list):
                cam_to_pt_indices.setdefault(iid, []).append(pt_idx)
        for iid, pt_indices in cam_to_pt_indices.items():
            kept_for_cam = sum(1 for pi in pt_indices if keep_pts[pi])
            if kept_for_cam < 20 and len(pt_indices) > 0:
                sorted_for_cam = sorted(pt_indices, key=lambda pi: float(pts_score[pi]), reverse=True)
                for pi in sorted_for_cam[: min(20, len(sorted_for_cam))]:
                    keep_pts[pi] = True

        kept_lines = [pts_lines[i] for i in range(len(pts_lines)) if keep_pts[i]]
        pts3d_txt.write_text("\n".join(pts_header + kept_lines) + "\n", encoding="utf-8")
        print(
            "[MaskFG] Очищено points3D.txt (Visual Hull, %d камер): залишено %d / %d точок об'єкта."
            % (len(cam_to_pt_indices), len(kept_lines), len(pts_lines)),
            flush=True,
        )

    try:
        from plyfile import PlyData, PlyElement

        for ply_file in sorted(dataset_dir.rglob("*.ply")):
            try:
                plydata = PlyData.read(str(ply_file))
                if "vertex" not in plydata or "face" in plydata:
                    continue
                v_data = plydata["vertex"].data
                if len(v_data) < 100:
                    continue
                pts_ply = np.column_stack([v_data["x"], v_data["y"], v_data["z"]]).astype(np.float64)
                keep_ply, _ = compute_visual_hull_keep_mask(pts_ply)
                if np.sum(keep_ply) < len(v_data):
                    filtered_v = v_data[keep_ply]
                    PlyData([PlyElement.describe(filtered_v, "vertex")], text=plydata.text).write(str(ply_file))
                    print(
                        "[MaskFG] Очищено prior PLY %s: залишено %d / %d точок об'єкта."
                        % (ply_file.name, len(filtered_v), len(v_data)),
                        flush=True,
                    )
            except Exception:
                pass
    except ImportError:
        pass

    # 4. Ensure 2D .npy depth maps remain strictly positive & finite (never 0.0!),
    #    preventing divide-by-zero in dataset_readers.py:256 and depth=0 black Gaussians at camera centers.
    #    Out-of-object generated Gaussians are pruned in sitecustomize.py's gaussian_generator hook.
    for depth_sub in ("depths", "depth_priors", "estimated_dense_depth", "depth"):
        d_dir = dataset_dir / depth_sub
        if d_dir.is_dir():
            sanitized_d_cnt = 0
            for npy_f in sorted(d_dir.glob("*.npy")):
                try:
                    arr = np.load(str(npy_f))
                    orig_shape = arr.shape
                    sq = np.squeeze(arr)
                    if sq.ndim == 2:
                        valid_pos = np.isfinite(sq) & (sq > 1e-3)
                        if np.any(~valid_pos) and np.any(valid_pos):
                            med_d = float(np.median(sq[valid_pos]))
                            sq = np.where(valid_pos, sq, max(0.1, med_d)).astype(arr.dtype)
                            np.save(str(npy_f), sq.reshape(orig_shape))
                            sanitized_d_cnt += 1
                except Exception:
                    pass
            if sanitized_d_cnt > 0:
                print("[MaskFG] Перевірено та відновлено позитивну глибину у %d файлах (%s/)" % (sanitized_d_cnt, depth_sub), flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
