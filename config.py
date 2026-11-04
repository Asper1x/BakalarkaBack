import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = Path(os.getenv("TRAIN_SCRIPT_PATH", BASE_DIR / "train_video.sh")).resolve()
UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", BASE_DIR / "uploads")).resolve()
JOBS_FILE = Path(os.getenv("JOBS_FILE", BASE_DIR / "jobs_state.json")).resolve()
GPU_STATS_FILE = Path(os.getenv("GPU_STATS_FILE", UPLOADS_DIR / "gpu_stats.csv")).resolve()

# Environment variables used by train_video.sh
SUGAR_ROOT = Path(os.getenv("SUGAR_ROOT", Path.home() / "2d-sugar")).expanduser()
DATASETS_ROOT = Path(os.getenv("DATASETS_ROOT", Path.home() / "datasets")).expanduser()
COLMAP_ENV = os.getenv("COLMAP_ENV", "colmap")
SUGAR_ENV = os.getenv("SUGAR_ENV", "2d-sugar")

# Slurm settings (defaults taken from job.slurm)
SLURM_PARTITION = os.getenv("SLURM_PARTITION", "dgx")
SLURM_CPUS_PER_TASK = int(os.getenv("SLURM_CPUS_PER_TASK", "20"))
SLURM_GRES = os.getenv("SLURM_GRES", "shard:30")

# Default pipeline parameters (matching job.slurm & train_video.sh)
DEFAULT_FPS = int(os.getenv("FPS", "6"))
DEFAULT_ITERATIONS = int(os.getenv("ITERATIONS", "20000"))
DEFAULT_RESOLUTION = int(os.getenv("RESOLUTION", "1"))
DEFAULT_VIDEO_RESOLUTION = os.getenv("VIDEO_RESOLUTION", "1920x1080")
DEFAULT_FILTER_BLURRY = os.getenv("FILTER_BLURRY", "1") == "1"
DEFAULT_MASK_FOREGROUND = os.getenv("MASK_FOREGROUND", "0") == "1"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
