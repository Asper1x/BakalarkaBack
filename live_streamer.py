import os
import queue
import struct
import threading
import time
from pathlib import Path
from typing import Optional

MAGIC_HEADER = 0x3D475301

class LiveStreamer:
    """
    Sub-samples top-K Gaussians on GPU directly into pinned host memory,
    dispatches via a lossy 1-slot queue, and delivers binary packets.
    """
    def __init__(
        self,
        job_id: Optional[str] = None,
        output_packet_path: Optional[str] = None,
        max_points: int = 65536,
        stream_interval: Optional[int] = None,
        streaming_flag_path: Optional[str] = None,
    ):
        self.job_id = job_id or os.getenv("JOB_ID", "default_job")
        self.max_points = max_points

        env_interval = os.getenv("STREAM_INTERVAL")
        if stream_interval is not None:
            self.stream_interval = stream_interval
        elif env_interval and env_interval.isdigit():
            self.stream_interval = int(env_interval)
        else:
            self.stream_interval = 15

        packet_p = output_packet_path or os.getenv("LIVE_STREAM_PACKET")
        self.output_packet_path = Path(packet_p) if packet_p else None

        flag_p = streaming_flag_path or os.getenv("STREAMING_FLAG")
        self.streaming_flag_path = Path(flag_p) if flag_p else None

        self.frame_queue = queue.Queue(maxsize=1)
        self.is_running = True

        # Pinned memory buffers
        self.pinned_xyz = None
        self.pinned_rgb = None
        self.stream = None
        self.has_torch = False

        self._init_torch()

        # Background worker for zero-blocking I/O
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _init_torch(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.pinned_xyz = torch.empty((self.max_points, 3), dtype=torch.float32, pin_memory=True)
                self.pinned_rgb = torch.empty((self.max_points, 3), dtype=torch.uint8, pin_memory=True)
                self.stream = torch.cuda.Stream()
                self.has_torch = True
        except Exception:
            self.has_torch = False

    def has_active_clients(self) -> bool:
        if self.streaming_flag_path:
            return self.streaming_flag_path.is_file()
        return False

    def step(self, iteration: int, gaussians, loss: float = 0.0):
        if iteration % self.stream_interval != 0:
            return
        if not self.has_active_clients():
            return
        if not self.has_torch:
            return

        import torch
        try:
            with torch.no_grad():
                # 1. Filter out dead/invisible Gaussians using opacity
                opacities = gaussians.get_opacity.squeeze(-1) # [N]

                # 2. Pick top-K visible points directly on GPU
                k = min(self.max_points, opacities.shape[0])
                if k <= 0:
                    return

                top_vals, top_indices = torch.topk(opacities, k, sorted=False)

                # 3. Extract positions [K, 3] in float32
                sampled_xyz = gaussians.get_xyz[top_indices].contiguous()

                # 4. Quantize spherical harmonics DC to uint8 RGB directly in CUDA
                # (0.28209479177387814 is SH C0 constant)
                rgb_float = 0.5 + 0.28209479177387814 * gaussians.get_features[top_indices, 0, :]
                sampled_rgb = (torch.clamp(rgb_float, 0.0, 1.0) * 255.0).to(torch.uint8).contiguous()

                # 5. Non-blocking asynchronous CUDA-to-Host transfer using pinned memory
                if self.stream is not None:
                    self.stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(self.stream):
                        self.pinned_xyz[:k].copy_(sampled_xyz, non_blocking=True)
                        self.pinned_rgb[:k].copy_(sampled_rgb, non_blocking=True)
                    self.stream.synchronize()
                else:
                    self.pinned_xyz[:k].copy_(sampled_xyz)
                    self.pinned_rgb[:k].copy_(sampled_rgb)

                xyz_bytes = self.pinned_xyz[:k].numpy().tobytes()
                rgb_bytes = self.pinned_rgb[:k].numpy().tobytes()

                # Push to single-slot queue (drop policy on backpressure)
                try:
                    self.frame_queue.put_nowait({
                        "iteration": iteration,
                        "loss": float(loss),
                        "count": k,
                        "xyz": xyz_bytes,
                        "rgb": rgb_bytes
                    })
                except queue.Full:
                    pass
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _worker_loop(self):
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                packet = self._pack_packet(
                    frame["iteration"],
                    frame["loss"],
                    frame["count"],
                    frame["xyz"],
                    frame["rgb"]
                )
                if self.output_packet_path:
                    tmp_path = self.output_packet_path.with_suffix(".tmp")
                    tmp_path.write_bytes(packet)
                    tmp_path.replace(self.output_packet_path)
            except Exception:
                pass
            finally:
                self.frame_queue.task_done()

    @staticmethod
    def _pack_packet(iteration: int, loss: float, count: int, xyz_bytes: bytes, rgb_bytes: bytes) -> bytes:
        header = struct.pack("<IIIf", MAGIC_HEADER, int(iteration), int(count), float(loss))
        return header + xyz_bytes + rgb_bytes

    def close(self):
        self.is_running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
