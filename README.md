# Video-to-3D Backend (FastAPI + Slurm)

Lightweight Python 3.12 + FastAPI backend that submits `train_video.sh` (FFmpeg -> COLMAP -> Metric3D -> 2D-SuGaR) jobs to **Slurm** (`sbatch`, `squeue`, `scancel`) based on `job.slurm`.

## How It Works
1. Client uploads a video via `POST /jobs`.
2. Backend saves the video to `uploads/<job_id>.mp4` and generates a per-job Slurm batch script `uploads/<job_id>.slurm` matching `job.slurm`:
   ```bash
   #!/bin/bash
   #SBATCH --job-name=2DSUGAR_<job_id>
   #SBATCH --partition=dgx
   #SBATCH --output=uploads/<job_id>_output.log
   #SBATCH --error=uploads/<job_id>_error.log
   #SBATCH --cpus-per-task=20
   #SBATCH --gres=shard:30
   ```
3. Backend submits the script with `sbatch --parsable` and tracks its Slurm job ID, Slurm state (`squeue` / `sacct`), exit code, and current pipeline stage (`1/5` to `5/5`).

## Setup & Run (on the Slurm login/submission node)

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
chmod +x train_video.sh

uvicorn main:app --host 0.0.0.0 --port 8000
```

### Optional Environment Variables
| Variable | Default | Description |
|---|---|---|
| `SLURM_PARTITION` | `dgx` | Slurm `--partition` |
| `SLURM_CPUS_PER_TASK` | `20` | Slurm `--cpus-per-task` |
| `SLURM_GRES` | `shard:30` | Slurm `--gres` GPU allocation |
| `SUGAR_ROOT` | `~/2d-sugar` | Path to 2D-SuGaR repository |
| `DATASETS_ROOT` | `~/datasets` | Path where datasets and logs are stored |
| `COLMAP_ENV` | `colmap` | Conda environment for COLMAP |
| `SUGAR_ENV` | `2d-sugar` | Conda environment for 2D-SuGaR |
| `FPS` | `6` | Default frame extraction FPS |
| `RESOLUTION` | `1` | Default resolution downsample factor |
| `ITERATIONS` | `30000` | Default training iterations |
| `VIDEO_RESOLUTION` | `1920x1080` | Default video frame scaling |

## API Endpoints

- `GET /health` — Check backend status, Slurm (`sbatch`) availability, and configured paths.
- `POST /jobs` — Upload a video (`multipart/form-data`) and submit a Slurm job.
  - Form fields: `video` (file, required), `name` (optional), `fps` (default `6`), `resolution` (default `1`), `iterations` (default `20000`), `video_resolution` (default `1920x1080`), `filter_blurry` (default `true`), `mask_foreground` (default `false`).
- `GET /jobs` — List all jobs and their current Slurm & pipeline status.
- `GET /jobs/{job_id}` — Get job status, `slurm_job_id`, `slurm_state`, progress stage (`1/5`–`5/5`), `has_cameras`, `has_sparse_cloud`, structured artifact keys (`step1_artifact`, `step2_artifact`, `companion_texture`, `companion_mtl`, `gaussian_ply`), and generated 3D model artifacts.
- `GET /jobs/{job_id}/gpu` — Get live GPU stats for this specific Slurm job.
- `GET /jobs/{job_id}/telemetry?step=100` — Get append-mode GPU history (`[timestamp, util_gpu, vram_mb, temp_c]`) and downsampled training loss/point curves (`step1_gaussian`, `step2_sugar`, `training`).
- `GET /jobs/{job_id}/cameras?normalize=true` — Get 3D camera frustums & poses from `cameras.json` (Stage 5) or COLMAP `cameras.txt`/`images.txt` (Stage 4), normalized to the 3D model bounding box.
- `GET /jobs/{job_id}/sparse-cloud?format=ply` — Download COLMAP sparse point cloud (`.ply` or `?format=json` array of `[x, y, z, r, g, b]`).
- `GET /jobs/{job_id}/frames` — List extracted/undistorted frames, available modalities (`rgb`, `depth`, `normal`), and per-frame Laplacian sharpness scores (`frame_quality`).
- `GET /jobs/{job_id}/frames/{modality}/{filename}` — Serve the RGB, depth, or normal image for a frame.
- `POST /jobs/{job_id}/crop` — Interactive 3D bounding-box crop (`{"artifact": "...", "min": [x,y,z], "max": [x,y,z]}`) saving `cropped_<name>.obj` / `.ply`.
- `GET /jobs/{job_id}/mesh-diff?format=json` — Geometric surface displacement stats (`mean`, `median`, `p95`, `max`) between Step 2 `.obj` and Step 1 `fuse_post.ply`, or vertex-colored `.ply` heatmap (`?format=ply`).
- `GET /jobs/{job_id}/splat` — Convert Gaussian `point_cloud.ply` into compact 32-byte-per-splat `.splat` binary format.
- `GET /jobs/{job_id}/metrics` — Quantitative quality report (`psnr`, `ssim`, `lpips`, frame/camera/gaussian/mesh vertex & face counts, `stage_durations_sec`).
- `GET /jobs/{job_id}/turntable` — Serve novel-view `turntable.mp4` video.
- `POST /jobs/{job_id}/decimate?target_faces=100000` — Simplify `.obj` mesh and save `decimated_{target_faces}.obj`.
- `GET /jobs/{job_id}/export.glb` — Export and download a self-contained binary glTF 2.0 (`.glb`) model with embedded texture and UVs.
- `GET /jobs/{job_id}/logs?tail=100` — Fetch combined pipeline & Slurm stdout/stderr logs.
- `GET /jobs/{job_id}/result` — Download the primary generated 3D mesh/model file (`sugarfine_*.obj`).
- `GET /jobs/{job_id}/files/{artifact_path}` — Download a specific output artifact.
- `DELETE /jobs/{job_id}` — Cancel a queued or running Slurm job via `scancel`.
- `WS /ws/jobs/{job_id}/stream` — Live in-memory binary WebSocket stream (Variant 3) pushing top-65,536 Gaussians (positions + RGB) at ~983 KB/packet during training. Runs at 0% overhead when no client is connected.
