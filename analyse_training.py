"""
Analyse Jetson training_mem.csv and produce a crash/resource report.
Usage: python3 analyse_training.py [--csv training_mem.csv]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def print_summary(df: pd.DataFrame) -> None:
    duration = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
    peak_ram = df["ram_used_mb"].max()
    peak_ram_pct = df["ram_pct"].max()
    peak_swap = df["swap_used_mb"].max()
    peak_gpu = df["gpu_load_pct"].max()
    peak_cpu = df["cpu_pct"].max()
    peak_temp = df["temp_tj-thermal"].max() if "temp_tj-thermal" in df.columns else None

    # Detect crash: log ends abruptly (last GPU load != 0, no graceful shutdown row)
    last_gpu = df["gpu_load_pct"].iloc[-1]
    last_ram = df["ram_used_mb"].iloc[-1]
    ram_at_crash = f"{last_ram:,.0f} MB ({df['ram_pct'].iloc[-1]:.1f}%)"

    print("=" * 65)
    print("  JETSON TRAINING RESOURCE REPORT")
    print("=" * 65)
    print(f"  Duration logged   : {duration}")
    print(f"  Total samples     : {len(df)}")
    print(f"  Peak RAM          : {peak_ram:,.0f} MB  ({peak_ram_pct:.1f}%)")
    print(f"  Peak Swap used    : {peak_swap:,.0f} MB")
    print(f"  Peak GPU load     : {peak_gpu:.0f}%")
    print(f"  Peak CPU          : {peak_cpu:.1f}%")
    if peak_temp:
        print(f"  Peak Tj (die temp): {peak_temp:.1f} °C")
    print(f"  RAM at log end    : {ram_at_crash}")
    print(f"  GPU at log end    : {last_gpu:.0f}%")
    print()

    # ── Crash diagnosis ───────────────────────────────────────
    issues = []
    if peak_ram_pct > 90:
        issues.append(f"CRITICAL — RAM hit {peak_ram_pct:.0f}% (likely OOM kill)")
    elif peak_ram_pct > 75:
        issues.append(f"WARNING  — RAM hit {peak_ram_pct:.0f}% (tight, swap may thrash)")
    if peak_swap > 500:
        issues.append(f"WARNING  — Swap used {peak_swap:,.0f} MB (indicates RAM pressure)")
    if peak_temp and peak_temp > 85:
        issues.append(f"CRITICAL — Tj reached {peak_temp:.1f}°C (thermal throttle/shutdown)")
    elif peak_temp and peak_temp > 75:
        issues.append(f"WARNING  — Tj reached {peak_temp:.1f}°C (approaching thermal limit)")

    # RAM trend at end — was it still climbing?
    last_n = df.tail(20)
    ram_slope = (last_n["ram_used_mb"].iloc[-1] - last_n["ram_used_mb"].iloc[0])
    if ram_slope > 200:
        issues.append(f"WARNING  — RAM was still growing (+{ram_slope:.0f} MB in last 20 samples)")

    if issues:
        print("  DIAGNOSIS:")
        for issue in issues:
            print(f"    • {issue}")
    else:
        print("  DIAGNOSIS: No crash indicators found — log may have ended normally.")
    print("=" * 65)


def plot(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    t = df["timestamp"]

    temp_cols = [c for c in df.columns if c.startswith("temp_")]

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle("Jetson Training Resource Monitor", fontsize=14, fontweight="bold")

    # ── 1. RAM & Swap ─────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, df["ram_used_mb"], label="RAM used (MB)", color="steelblue", linewidth=1.2)
    ax.plot(t, df["swap_used_mb"], label="Swap used (MB)", color="orange", linewidth=1.2, linestyle="--")
    ax.axhline(df["ram_total_mb"].iloc[0], color="red", linestyle=":", linewidth=0.8, label="RAM total")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("RAM & Swap Usage")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 2. RAM % ──────────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(t, df["ram_pct"], alpha=0.4, color="steelblue")
    ax.plot(t, df["ram_pct"], color="steelblue", linewidth=1.0)
    ax.axhline(90, color="red", linestyle="--", linewidth=0.8, label="90% danger line")
    ax.axhline(75, color="orange", linestyle="--", linewidth=0.8, label="75% warning line")
    ax.set_ylabel("RAM %")
    ax.set_ylim(0, 105)
    ax.set_title("RAM Utilisation %")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 3. GPU & CPU ──────────────────────────────────────────
    ax = axes[2]
    ax.plot(t, df["gpu_load_pct"], label="GPU load %", color="green", linewidth=1.0)
    ax.plot(t, df["cpu_pct"], label="CPU %", color="purple", linewidth=1.0, alpha=0.7)
    ax.set_ylabel("Load %")
    ax.set_ylim(0, 105)
    ax.set_title("GPU & CPU Load")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 4. Temperatures ───────────────────────────────────────
    ax = axes[3]
    colors = plt.cm.tab10.colors
    for i, col in enumerate(temp_cols):
        label = col.replace("temp_", "").replace("-thermal", "")
        ax.plot(t, df[col], label=label, linewidth=1.0, color=colors[i % len(colors)])
    ax.axhline(85, color="red", linestyle="--", linewidth=0.8, label="85°C shutdown")
    ax.axhline(75, color="orange", linestyle="--", linewidth=0.8, label="75°C warning")
    ax.set_ylabel("°C")
    ax.set_title("Temperatures")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # ── X-axis formatting ─────────────────────────────────────
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    out_path = out_dir / "jetson_training_report.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Graph saved → {out_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse Jetson training resource log")
    parser.add_argument("--csv", default="training_mem.csv", help="Path to CSV log file")
    parser.add_argument("--out", default="training_report", help="Output directory for graphs")
    args = parser.parse_args()

    csv_path = args.csv
    if not Path(csv_path).exists():
        print(f"ERROR: {csv_path} not found.", file=sys.stderr)
        sys.exit(1)

    df = load(csv_path)
    print_summary(df)
    plot(df, Path(args.out))


if __name__ == "__main__":
    main()
