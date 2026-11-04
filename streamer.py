import asyncio
import os
import struct
from pathlib import Path
from typing import Dict, Set
from fastapi import WebSocket

MAGIC_HEADER = 0x3D475301  # Protocol identifier header

class StreamHub:
    def __init__(self):
        # job_id -> set of active WebSockets
        self._clients: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def register_client(self, job_id: str, websocket: WebSocket):
        async with self._lock:
            if job_id not in self._clients:
                self._clients[job_id] = set()
            self._clients[job_id].add(websocket)
            self._write_active_client_flag(job_id, len(self._clients[job_id]))

    async def unregister_client(self, job_id: str, websocket: WebSocket):
        async with self._lock:
            if job_id in self._clients:
                self._clients[job_id].discard(websocket)
                count = len(self._clients[job_id])
                if count == 0:
                    del self._clients[job_id]
                self._write_active_client_flag(job_id, count)

    def has_active_clients(self, job_id: str) -> bool:
        return job_id in self._clients and len(self._clients[job_id]) > 0

    def get_active_client_count(self, job_id: str) -> int:
        return len(self._clients.get(job_id, ()))

    def _write_active_client_flag(self, job_id: str, count: int):
        import config
        flag_file = config.UPLOADS_DIR / f"{job_id}.streaming_clients"
        try:
            if count > 0:
                flag_file.write_text(str(count), encoding="utf-8")
            else:
                if flag_file.exists():
                    flag_file.unlink(missing_ok=True)
        except Exception:
            pass

    async def broadcast_binary(self, job_id: str, packet: bytes):
        async with self._lock:
            sockets = list(self._clients.get(job_id, []))

        if not sockets:
            return

        dead_sockets = []
        for ws in sockets:
            try:
                await ws.send_bytes(packet)
            except Exception:
                dead_sockets.append(ws)

        if dead_sockets:
            async with self._lock:
                for ws in dead_sockets:
                    if job_id in self._clients:
                        self._clients[job_id].discard(ws)
                if job_id in self._clients and len(self._clients[job_id]) == 0:
                    del self._clients[job_id]
                self._write_active_client_flag(job_id, len(self._clients.get(job_id, ())))

    @staticmethod
    def pack_binary_packet(iteration: int, loss: float, count: int, xyz_bytes: bytes, rgb_bytes: bytes) -> bytes:
        """
        Packs header + position buffer + color buffer:
        Header (16B):
          0..3   magic: uint32 (0x3D475301)
          4..7   iteration: uint32
          8..11  point_count: uint32 (K)
          12..15 loss: float32
        Payload:
          16..(16 + K*12)  positions: float32[K][3]
          (16 + K*12)..END colors: uint8[K][3]
        """
        header = struct.pack("<IIIf", MAGIC_HEADER, int(iteration), int(count), float(loss))
        return header + xyz_bytes + rgb_bytes

    @staticmethod
    def unpack_binary_header(header_bytes: bytes):
        if len(header_bytes) < 16:
            return None
        magic, iteration, count, loss = struct.unpack("<IIIf", header_bytes[:16])
        return {
            "magic": magic,
            "iteration": iteration,
            "count": count,
            "loss": loss
        }

stream_hub = StreamHub()
