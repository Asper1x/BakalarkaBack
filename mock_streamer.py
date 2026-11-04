"""
Mock Live Stream Generator for testing Variant 3 live streaming.
Generates synthetic Gaussian point clouds at 5 Hz and writes binary packets
to uploads/{job_id}_live_stream.bin matching the protocol in streamer.py.
"""
import argparse
import math
import struct
import time
from pathlib import Path

MAGIC_HEADER = 0x3D475301

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=str, default="test_job")
    parser.add_argument("--count", type=int, default=65536)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)

    packet_path = uploads_dir / f"{args.job_id}_live_stream.bin"
    flag_path = uploads_dir / f"{args.job_id}.streaming_clients"

    print(f"Mock live streamer started for job '{args.job_id}'")
    print(f"Packet destination: {packet_path}")
    print(f"Active clients flag: {flag_path} (exists={flag_path.exists()})")

    K = args.count
    sleep_interval = 1.0 / args.fps

    for it in range(1, args.iterations + 1):
        if flag_path.exists():
            # Generate synthetic point cloud (sphere with pulsating radius)
            radius = 1.0 + 0.2 * math.sin(it * 0.1)
            loss = max(0.005, 0.5 * math.exp(-it / 200.0) + 0.01 * math.sin(it))

            pos_bytes = bytearray(K * 12)
            col_bytes = bytearray(K * 3)

            # Pack sphere points with color
            phi_step = 2.3999632  # Golden ratio angle
            for i in range(min(K, 4096)):  # Synthesize first batch, fill rest
                y = 1.0 - (i / float(K - 1)) * 2.0
                rad_at_y = math.sqrt(max(0.0, 1.0 - y * y)) * radius
                theta = phi_step * i
                x = math.cos(theta) * rad_at_y
                z = math.sin(theta) * rad_at_y

                struct.pack_into("<fff", pos_bytes, i * 12, x, y * radius, z)

                r = int(min(255, max(0, (x / radius * 0.5 + 0.5) * 255)))
                g = int(min(255, max(0, (y / radius * 0.5 + 0.5) * 255)))
                b = int(min(255, max(0, (z / radius * 0.5 + 0.5) * 255)))
                col_bytes[i * 3] = r
                col_bytes[i * 3 + 1] = g
                col_bytes[i * 3 + 2] = b

            header = struct.pack("<IIIf", MAGIC_HEADER, it, K, float(loss))
            packet = header + bytes(pos_bytes) + bytes(col_bytes)

            tmp_path = packet_path.with_suffix(".tmp")
            tmp_path.write_bytes(packet)
            tmp_path.replace(packet_path)
            print(f"[Iter {it}] Dispatched {len(packet)} bytes (loss={loss:.4f})")
        else:
            print(f"[Iter {it}] No active clients watching (0% overhead mode)")

        time.sleep(sleep_interval)

if __name__ == "__main__":
    main()
