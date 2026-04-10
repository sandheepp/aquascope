#!/usr/bin/env python3
"""
Jetson Orin Nano Resource Monitor
Logs CPU, GPU, RAM, swap, and thermal data to a CSV file.
Run this in a separate terminal while training:
    python3 jetson_monitor.py --interval 2 --output training_log.csv
"""

import argparse
import csv
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path


def read_file(path: str) -> str:
    """Read a sysfs/procfs file, return stripped content or empty string."""
    try:
        return Path(path).read_bytes().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# GPU (Jetson uses integrated GPU exposed via sysfs, not nvidia-smi)
# ---------------------------------------------------------------------------
GPU_LOAD_PATHS = [
    "/sys/devices/gpu.0/load",                              # Jetson Orin
    "/sys/devices/platform/gpu.0/load",                     # Older Jetsons
    "/sys/devices/platform/17000000.gpu/load",              # Some Orin variants
    "/sys/devices/platform/17000000.ga10b/load",            # Orin Nano/NX
]

GPU_FREQ_PATHS = [
    "/sys/devices/gpu.0/devfreq/17000000.gpu/cur_freq",
    "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq",
    "/sys/devices/platform/17000000.ga10b/devfreq/17000000.ga10b/cur_freq",
]


def find_working_path(candidates: list[str]) -> str | None:
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


GPU_LOAD_PATH = find_working_path(GPU_LOAD_PATHS)
GPU_FREQ_PATH = find_working_path(GPU_FREQ_PATHS)


def get_gpu_usage() -> dict:
    """Return GPU load (%) and current frequency (MHz)."""
    load = -1.0
    freq_mhz = -1.0

    if GPU_LOAD_PATH:
        raw = read_file(GPU_LOAD_PATH)
        if raw:
            # Value is 0-1000 on Jetson (divide by 10 for %)
            load = int(raw) / 10.0

    if GPU_FREQ_PATH:
        raw = read_file(GPU_FREQ_PATH)
        if raw:
            freq_mhz = int(raw) / 1_000_000  # Hz -> MHz

    return {"gpu_load_pct": load, "gpu_freq_mhz": freq_mhz}


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------
_prev_idle = 0
_prev_total = 0


def get_cpu_usage() -> float:
    """Return overall CPU usage % since last call (from /proc/stat)."""
    global _prev_idle, _prev_total
    line = read_file("/proc/stat").split("\n")[0]  # "cpu  user nice system idle ..."
    parts = line.split()
    vals = list(map(int, parts[1:]))
    idle = vals[3] + vals[4]  # idle + iowait
    total = sum(vals)

    d_idle = idle - _prev_idle
    d_total = total - _prev_total
    _prev_idle, _prev_total = idle, total

    if d_total == 0:
        return 0.0
    return round((1.0 - d_idle / d_total) * 100, 1)


# ---------------------------------------------------------------------------
# Memory & Swap
# ---------------------------------------------------------------------------
def get_memory() -> dict:
    """Return RAM and swap usage from /proc/meminfo."""
    info = {}
    for line in read_file("/proc/meminfo").split("\n"):
        parts = line.split()
        if len(parts) >= 2:
            info[parts[0].rstrip(":")] = int(parts[1])  # kB

    mem_total = info.get("MemTotal", 1)
    mem_avail = info.get("MemAvailable", 0)
    mem_used = mem_total - mem_avail

    swap_total = info.get("SwapTotal", 0)
    swap_free = info.get("SwapFree", 0)
    swap_used = swap_total - swap_free

    return {
        "ram_used_mb": round(mem_used / 1024, 1),
        "ram_total_mb": round(mem_total / 1024, 1),
        "ram_pct": round(mem_used / mem_total * 100, 1),
        "swap_used_mb": round(swap_used / 1024, 1),
        "swap_total_mb": round(swap_total / 1024, 1),
    }


# ---------------------------------------------------------------------------
# Thermal zones
# ---------------------------------------------------------------------------
THERMAL_BASE = "/sys/devices/virtual/thermal/"


def get_temperatures() -> dict:
    """Read all thermal zones and return as {zone_name: temp_C}."""
    temps = {}
    base = Path(THERMAL_BASE)
    if not base.exists():
        return temps

    for zone_dir in sorted(base.glob("thermal_zone*")):
        name = read_file(str(zone_dir / "type")) or zone_dir.name
        raw = read_file(str(zone_dir / "temp"))
        if raw:
            temps[name] = int(raw) / 1000.0  # millidegrees -> °C
    return temps


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def build_header(thermal_keys: list[str]) -> list[str]:
    return [
        "timestamp",
        "cpu_pct",
        "ram_used_mb",
        "ram_total_mb",
        "ram_pct",
        "swap_used_mb",
        "swap_total_mb",
        "gpu_load_pct",
        "gpu_freq_mhz",
    ] + [f"temp_{k}" for k in thermal_keys]


def monitor(interval: float, output: str, duration: float | None):
    # Warm up CPU delta calculation
    get_cpu_usage()
    time.sleep(0.1)

    # Discover thermal zones once
    thermal = get_temperatures()
    thermal_keys = sorted(thermal.keys())

    header = build_header(thermal_keys)
    file_exists = os.path.exists(output)

    stop = False

    def handle_signal(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    start_time = time.time()

    print(f"Logging to: {output}  (Ctrl+C to stop)")
    print(f"Interval:   {interval}s")
    if GPU_LOAD_PATH:
        print(f"GPU sysfs:  {GPU_LOAD_PATH}")
    else:
        print("WARNING: No GPU load sysfs path found — GPU load will show -1")
    print("-" * 70)

    with open(output, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists or os.path.getsize(output) == 0:
            writer.writerow(header)

        while not stop:
            if duration and (time.time() - start_time) > duration:
                break

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cpu = get_cpu_usage()
            mem = get_memory()
            gpu = get_gpu_usage()
            thermal = get_temperatures()

            row = [
                now,
                cpu,
                mem["ram_used_mb"],
                mem["ram_total_mb"],
                mem["ram_pct"],
                mem["swap_used_mb"],
                mem["swap_total_mb"],
                gpu["gpu_load_pct"],
                gpu["gpu_freq_mhz"],
            ] + [thermal.get(k, -1) for k in thermal_keys]

            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

            # Print live summary
            print(
                f"[{now}]  "
                f"CPU: {cpu:5.1f}%  |  "
                f"RAM: {mem['ram_used_mb']:,.0f}/{mem['ram_total_mb']:,.0f}MB ({mem['ram_pct']:.0f}%)  |  "
                f"GPU: {gpu['gpu_load_pct']:.0f}%  |  "
                f"Swap: {mem['swap_used_mb']:,.0f}/{mem['swap_total_mb']:,.0f}MB"
            )

            time.sleep(interval)

    print(f"\nDone. Log saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jetson Orin Nano resource monitor")
    parser.add_argument("-i", "--interval", type=float, default=2.0, help="Sampling interval in seconds (default: 2)")
    parser.add_argument("-o", "--output", type=str, default="training_log.csv", help="Output CSV path (default: training_log.csv)")
    parser.add_argument("-d", "--duration", type=float, default=None, help="Stop after N seconds (default: run until Ctrl+C)")
    args = parser.parse_args()

    monitor(args.interval, args.output, args.duration)