import asyncio
from pathlib import Path
import shutil
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

import config
from jobs import JobStatus, job_manager
from streamer import stream_hub

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class CropRequest(BaseModel):
    artifact: Optional[str] = None
    min: list[float]
    max: list[float]
    normalized: Optional[bool] = None

app = FastAPI(
    title="Video-to-3D Slurm Backend API",
    description="FastAPI backend submitting train_video.sh (COLMAP + 2D-SuGaR) jobs to Slurm.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "script_exists": config.SCRIPT_PATH.is_file(),
        "sbatch_available": shutil.which("sbatch") is not None,
        "slurm_partition": config.SLURM_PARTITION,
        "slurm_cpus_per_task": config.SLURM_CPUS_PER_TASK,
        "slurm_gres": config.SLURM_GRES,
        "sugar_root": str(config.SUGAR_ROOT),
        "datasets_root": str(config.DATASETS_ROOT),
    }


@app.get("/gpu")
async def get_gpu_stats():
    """Return current GPU utilization, VRAM usage (MB and %), and temperature."""
    return await job_manager.get_gpu_stats()


@app.get("/datasets")
async def list_datasets():
    """Scan DATASETS_ROOT and list all valid datasets with their readiness level."""
    return job_manager.get_datasets_list()


@app.get("/datasets/{dataset_name}/thumbnail")
async def get_dataset_thumbnail(dataset_name: str):
    """Return a resized preview image (thumbnail) for an existing dataset."""
    safe_name = dataset_name.strip()
    if not job_manager.NAME_PATTERN.match(safe_name):
        raise HTTPException(status_code=400, detail="Invalid dataset name.")

    root = config.DATASETS_ROOT.resolve()
    target = (root / safe_name).resolve()
    if not str(target).startswith(str(root)) or not target.is_dir():
        raise HTTPException(status_code=404, detail="Dataset not found.")

    img_dir = None
    for cand in ["images", "input"]:
        d = target / cand
        if d.is_dir():
            img_dir = d
            break
    if img_dir is None:
        img_dir = target

    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    preview_file = None
    for f in sorted(img_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in image_exts:
            preview_file = f
            break

    if not preview_file:
        raise HTTPException(status_code=404, detail="No preview image found in dataset.")

    # Generate or return cached thumbnail
    thumb_path = config.UPLOADS_DIR / f"thumb_{safe_name}.jpg"
    if thumb_path.is_file() and thumb_path.stat().st_mtime >= preview_file.stat().st_mtime:
        return FileResponse(thumb_path, media_type="image/jpeg")

    try:
        from PIL import Image
        with Image.open(preview_file) as im:
            im = im.convert("RGB")
            im.thumbnail((240, 240))
            im.save(thumb_path, "JPEG", quality=85)
        return FileResponse(thumb_path, media_type="image/jpeg")
    except Exception:
        # Fallback to serving the raw image directly
        media = "image/png" if preview_file.suffix.lower() == ".png" else "image/jpeg"
        return FileResponse(preview_file, media_type=media)


@app.post("/jobs", status_code=202)
async def create_job(
    input_mode: str = Form("video", description="'video' or 'dataset'"),
    video: Optional[UploadFile] = File(None, description="Input video file (required if input_mode='video')"),
    dataset_name: Optional[str] = Form(None, description="Existing dataset name in DATASETS_ROOT (required if input_mode='dataset')"),
    name: Optional[str] = Form(None, description="Optional project name (letters, digits, '_', '-')"),
    fps: int = Form(config.DEFAULT_FPS, ge=1, le=60, description="Frames per second to extract"),
    iterations: int = Form(config.DEFAULT_ITERATIONS, ge=100, le=100_000, description="Training iterations"),
    resolution: int = Form(config.DEFAULT_RESOLUTION, ge=1, le=8, description="Resolution downsample factor (-r)"),
    video_resolution: str = Form(config.DEFAULT_VIDEO_RESOLUTION, description="Frame scale resolution, e.g. 1920x1080"),
    filter_blurry: bool = Form(config.DEFAULT_FILTER_BLURRY, description="Laplacian sharpness frame filtering in Stage 1"),
    mask_foreground: bool = Form(config.DEFAULT_MASK_FOREGROUND, description="Optional foreground masking (rembg) in Stage 1"),
    stream_interval: int = Form(15, ge=1, le=1000, description="Iterations between live 3DGS stream packet frames"),
    scene_type: str = Form("object", description="Target scene type: 'object' (small asset) or 'building' (architectural facade)"),
    refinement_mode: str = Form("full", description="SuGaR refinement mode: 'skip' (0 iters), 'fast' (2000), 'balanced' (7000), 'full' (15000)"),
    watertight_mesh: bool = Form(False, description="Close all mesh holes to guarantee 100% watertight 2-manifold surface"),
):
    if input_mode not in ("video", "dataset"):
        raise HTTPException(status_code=400, detail="input_mode must be 'video' or 'dataset'.")

    saved_path = None
    if input_mode == "video":
        if not video or not video.filename:
            raise HTTPException(status_code=400, detail="Video file is required when input_mode is 'video'.")

        ext = Path(video.filename).suffix.lower()
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension '{ext}'. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
            )

        temp_filename = f"upload_{Path(video.filename).name}"
        saved_path = config.UPLOADS_DIR / temp_filename
        with saved_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
    elif input_mode == "dataset":
        if not dataset_name:
            raise HTTPException(status_code=400, detail="dataset_name is required when input_mode is 'dataset'.")

    try:
        job = await job_manager.submit_job(
            video_path=saved_path,
            input_mode=input_mode,
            dataset_name=dataset_name,
            project_name=name,
            fps=fps,
            iterations=iterations,
            resolution=resolution,
            video_resolution=video_resolution,
            filter_blurry=filter_blurry,
            mask_foreground=mask_foreground,
            stream_interval=stream_interval,
            scene_type=scene_type,
            refinement_mode=refinement_mode,
            watertight_mesh=watertight_mesh,
        )
        return job.to_dict()
    except (ValueError, FileNotFoundError) as exc:
        if saved_path and saved_path.exists():
            saved_path.unlink(missing_ok=True)
        code = 404 if isinstance(exc, FileNotFoundError) else 400
        raise HTTPException(status_code=code, detail=str(exc))
    except RuntimeError as exc:
        if saved_path and saved_path.exists():
            saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/jobs")
async def list_jobs():
    jobs = await job_manager.list_jobs()
    return [job.to_dict() for job in jobs]


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job.to_dict()


@app.get("/jobs/{job_id}/gpu")
async def get_job_gpu_stats(job_id: str):
    """Return GPU utilization and VRAM stats from the Slurm node running this job."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return await job_manager.get_gpu_stats(job=job)


@app.get("/jobs/{job_id}/logs", response_class=PlainTextResponse)
async def get_job_logs(
    job_id: str,
    tail: Optional[int] = Query(None, ge=1, le=10000, description="Return only the last N lines"),
):
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job_manager.get_logs(job, tail_lines=tail)


@app.get("/jobs/{job_id}/cameras")
async def get_job_cameras(
    job_id: str,
    normalize: bool = Query(True, description="Normalize camera positions to match the 3D model bounding box"),
):
    """Return 3D camera frustums and trajectory poses from Stage 5 cameras.json or Stage 4 COLMAP TXT."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    cameras = job_manager.get_cameras(job, normalize=normalize)
    if not cameras:
        raise HTTPException(status_code=404, detail="Camera data is not available for this job yet.")
    return cameras


@app.get("/jobs/{job_id}/sparse-cloud")
async def get_job_sparse_cloud(
    job_id: str,
    format: str = Query("ply", pattern="^(ply|json)$", description="Output format: 'ply' or 'json'"),
):
    """Return COLMAP sparse point cloud from Stage 5 input.ply or Stage 4 points3D.txt."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if format == "json":
        points = job_manager.parse_points3d_txt(job)
        if not points:
            raise HTTPException(status_code=404, detail="Sparse point cloud is not available yet.")
        return points

    ply_path = job_manager.get_sparse_cloud_ply_path(job)
    if not ply_path or not ply_path.is_file():
        raise HTTPException(status_code=404, detail="Sparse point cloud is not available yet.")

    return FileResponse(
        path=ply_path,
        filename="sparse_cloud.ply",
        media_type="application/octet-stream",
    )


@app.get("/jobs/{job_id}/frames")
async def get_job_frames(job_id: str):
    """Return list of extracted/undistorted frame filenames and available modalities (rgb, depth, normal)."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job_manager.get_frames_info(job)


@app.get("/jobs/{job_id}/frames/{modality}/{filename}")
async def get_job_frame_image(
    job_id: str,
    modality: str,
    filename: str,
    thumbnail: bool = Query(False, description="Whether to serve a lightweight preview thumbnail"),
    max_dim: Optional[int] = Query(None, ge=64, le=3840, description="Max dimension for downscaling"),
):
    """Return the image file for a specific frame and modality (rgb, depth, normal)."""
    if modality not in ("rgb", "depth", "normal"):
        raise HTTPException(status_code=400, detail="Modality must be one of: rgb, depth, normal.")

    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    target_dim = max_dim
    if thumbnail and not target_dim:
        target_dim = 480

    file_path = await asyncio.to_thread(
        job_manager.resolve_frame_file,
        job=job,
        modality=modality,
        filename=filename,
        max_dim=target_dim,
    )
    if not file_path or not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"Frame '{filename}' for modality '{modality}' not found.")

    suffix = file_path.suffix.lower()
    media_type = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    if suffix == ".webp":
        media_type = "image/webp"

    return FileResponse(
        path=file_path,
        media_type=media_type,
        headers={
            "Cache-Control": "public, max-age=86400, immutable",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.get("/jobs/{job_id}/telemetry")
async def get_job_telemetry(
    job_id: str,
    step: int = Query(100, ge=1, le=5000, description="Downsample interval for training iterations"),
):
    """Return GPU time-series history and downsampled training loss/point curves for Step 1 & Step 2."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job_manager.get_telemetry(job, downsample_step=step)


@app.post("/jobs/{job_id}/crop")
async def crop_job_mesh(job_id: str, body: CropRequest):
    """Crop a .obj or .ply artifact using a 3D bounding box [min, max] and save cropped_<name>."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    try:
        return job_manager.crop_artifact(
            job=job,
            min_bounds=body.min,
            max_bounds=body.max,
            artifact=body.artifact,
            normalized=body.normalized,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/jobs/{job_id}/mesh-diff")
async def get_job_mesh_diff(
    job_id: str,
    format: str = Query("json", pattern="^(json|ply)$", description="Return 'json' stats or vertex-colored 'ply' heatmap"),
):
    """Compute geometric surface displacement between Step 2 .obj and Step 1 fuse_post.ply."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    try:
        diff_info = job_manager.compute_mesh_diff(job)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if format == "ply":
        output_root = (config.SUGAR_ROOT / "output").resolve()
        heatmap_path = (output_root / diff_info["heatmap_artifact"]).resolve()
        return FileResponse(
            path=heatmap_path,
            filename="mesh_diff_heatmap.ply",
            media_type="application/octet-stream",
        )
    return diff_info


@app.get("/jobs/{job_id}/splat")
async def get_job_splat(job_id: str):
    """Convert raw Gaussian Splatting point_cloud.ply into compact 32-byte-per-splat .splat binary format."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    splat_path = job_manager.get_or_create_splat(job)
    if not splat_path or not splat_path.is_file():
        raise HTTPException(status_code=404, detail="Gaussian point cloud not available for this job.")
    return FileResponse(
        path=splat_path,
        filename=f"{job.name}.splat",
        media_type="application/octet-stream",
    )


@app.get("/jobs/{job_id}/metrics")
async def get_job_metrics(job_id: str):
    """Return quantitative evaluation metrics (PSNR, SSIM, LPIPS), mesh/point counts, and stage durations."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job_manager.get_metrics(job)


@app.get("/jobs/{job_id}/turntable")
async def get_job_turntable(job_id: str):
    """Serve the rendered novel-view turntable MP4 video for this job."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    turntable_path = job_manager.get_turntable_path(job)
    if not turntable_path:
        raise HTTPException(status_code=404, detail="Turntable video is not available for this job.")
    return FileResponse(
        path=turntable_path,
        filename=f"{job.name}_turntable.mp4",
        media_type="video/mp4",
    )


@app.post("/jobs/{job_id}/decimate")
async def decimate_job_mesh(
    job_id: str,
    target_faces: int = Query(100000, ge=1000, le=2000000, description="Target triangle count after decimation"),
    artifact: Optional[str] = Query(None, description="Optional specific .obj artifact to decimate"),
):
    """Simplify the primary .obj mesh to target_faces and save decimated_{target_faces}.obj."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    try:
        return job_manager.decimate_mesh(job, target_faces=target_faces, artifact=artifact)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/jobs/{job_id}/export.glb")
async def export_job_glb(job_id: str):
    """Export and download a self-contained binary glTF 2.0 (.glb) model including textures and UVs."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    glb_path = job_manager.export_glb(job)
    if not glb_path or not glb_path.is_file():
        raise HTTPException(status_code=404, detail="Primary 3D mesh is not available for .glb export.")
    return FileResponse(
        path=glb_path,
        filename=f"{job.name}.glb",
        media_type="model/gltf-binary",
    )


@app.get("/jobs/{job_id}/result")
async def download_primary_result(job_id: str):
    """Download the main generated 3D model file (prefers mesh output, falls back to point cloud)."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=409, detail=f"Job is not completed (current status: {job.status}).")

    artifacts = job.list_artifacts()
    if not artifacts:
        raise HTTPException(status_code=404, detail="No generated 3D model files found for this job.")

    output_root = (config.SUGAR_ROOT / "output").resolve()
    target_path = (output_root / artifacts[0]).resolve()
    return FileResponse(
        path=target_path,
        filename=target_path.name,
        media_type="application/octet-stream",
    )


@app.get("/jobs/{job_id}/files/{artifact_path:path}")
async def download_artifact(job_id: str, artifact_path: str):
    """Download a specific file from the job's output artifacts list."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    if artifact_path not in job.list_artifacts():
        raise HTTPException(status_code=404, detail="Artifact not found for this job.")

    output_root = (config.SUGAR_ROOT / "output").resolve()
    target_path = (output_root / artifact_path).resolve()

    if not target_path.is_relative_to(output_root) or not target_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    return FileResponse(
        path=target_path,
        filename=target_path.name,
        media_type="application/octet-stream",
    )


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    job = await job_manager.cancel_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job.to_dict()


@app.websocket("/ws/jobs/{job_id}/stream")
async def websocket_stream_endpoint(websocket: WebSocket, job_id: str):
    """
    Live binary WebSocket stream for 3D Gaussian Splatting view during training.
    Pushes ~983 KB binary packets: [Header: 16B | Pos: K*12B | RGB: K*3B]
    """
    await websocket.accept()
    job = await job_manager.get_job(job_id)
    if not job:
        await websocket.close(code=4004, reason="Job not found")
        return

    await stream_hub.register_client(job_id, websocket)

    import asyncio
    import struct
    from streamer import MAGIC_HEADER

    packet_path = job.live_packet_path
    last_mtime = 0.0
    last_iteration = -1

    async def sender_task():
        nonlocal last_mtime, last_iteration
        while True:
            try:
                if packet_path.is_file():
                    stat = packet_path.stat()
                    if stat.st_mtime > last_mtime:
                        packet_data = packet_path.read_bytes()
                        if len(packet_data) >= 16:
                            magic, it, count, loss = struct.unpack("<IIIf", packet_data[:16])
                            if magic == MAGIC_HEADER and it != last_iteration and count > 0:
                                last_mtime = stat.st_mtime
                                last_iteration = it
                                await websocket.send_bytes(packet_data)
                await asyncio.sleep(0.03)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.04)

    task = asyncio.create_task(sender_task())

    try:
        while True:
            # Keep socket open and respond to client messages / pings
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await stream_hub.unregister_client(job_id, websocket)

