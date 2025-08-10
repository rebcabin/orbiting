#! python
import argparse
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

SECONDS_PER_YEAR = 365.25 * 24 * 3600.0

def main():
    parser = argparse.ArgumentParser(description="Animate orbit and element evolution with 1PN energy and progress.")
    parser.add_argument("--csv", default="orbits.csv", help="Path to CSV file")
    parser.add_argument("--out", default="orbit_animation.mp4", help="Output video (mp4 or gif). Empty to skip saving.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--trail", type=int, default=400, help="Trail length in samples")
    parser.add_argument("--step", type=int, default=5, help="Use every Nth row")
    parser.add_argument("--duration", type=float, default=None, help="Limit to first D seconds (simulation time)")
    parser.add_argument("--max-frames", type=int, default=2500, help="Cap total frames by uniform subsampling")
    parser.add_argument("--save-dpi", type=int, default=120, help="DPI when saving video")
    parser.add_argument("--display-seconds", type=int, default=12, help="Seconds to keep window open")
    args = parser.parse_args()

    print("[1/6] Reading CSV...", flush=True)
    df = pd.read_csv(args.csv)

    print("[2/6] Preparing data...", flush=True)
    if args.duration is not None:
        df = df[df["t"] <= args.duration].reset_index(drop=True)

    df = df.iloc[::max(1, args.step)].reset_index(drop=True)
    if args.max_frames is not None and len(df) > args.max_frames:
        idx = np.linspace(0, len(df) - 1, args.max_frames).astype(int)
        df = df.iloc[idx].reset_index(drop=True)

    required = {"t","xN","yN","xGR","yGR","eN","eGR","EN","EGR","E1PN_GR"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns in CSV: {missing}")
        sys.exit(1)

    # Extract arrays
    t_sec = df["t"].to_numpy()
    t_yr  = t_sec / SECONDS_PER_YEAR
    xN, yN = df["xN"].to_numpy(), df["yN"].to_numpy()
    xG, yG = df["xGR"].to_numpy(), df["yGR"].to_numpy()
    eN, eG = df["eN"].to_numpy(), df["eGR"].to_numpy()
    EN     = df["EN"].to_numpy()
    E1PN   = df["E1PN_GR"].to_numpy()

    EN0   = float(EN[0])
    E1PN0 = float(E1PN[0])
    dEN_rel   = (EN   - EN0)   / abs(EN0)
    dE1PN_rel = (E1PN - E1PN0) / abs(E1PN0)

    eN0, eG0 = float(eN[0]), float(eG[0])
    ecc_text = f"(eN≈{eN0:.3f}, eGR≈{eG0:.3f})"

    print("[3/6] Building figure and static elements...", flush=True)
    fig, (ax_orbit, ax_metrics) = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={"height_ratios": [3, 2]})
    fig.set_size_inches(*(fig.get_size_inches() * 0.85), forward=True)

    # ORBIT: static paths
    ax_orbit.plot(xN, yN, color="#1f77b4", lw=1.0, alpha=0.5, label="Newtonian", antialiased=False)
    ax_orbit.plot(xG, yG, color="#d62728", lw=1.0, alpha=0.7, label="GR (1PN)", antialiased=False)
    ax_orbit.scatter(0.0, 0.0, s=60, color="black", marker="o", label="SMBH", zorder=5)

    # Moving heads and trails
    headN, = ax_orbit.plot([], [], marker="o", color="#1f77b4", ms=5, zorder=6, antialiased=False)
    headG, = ax_orbit.plot([], [], marker="o", color="#d62728", ms=5, zorder=6, antialiased=False)
    trailN, = ax_orbit.plot([], [], color="#1f77b4", lw=2.0, alpha=0.9, antialiased=False)
    trailG, = ax_orbit.plot([], [], color="#d62728", lw=2.0, alpha=0.9, antialiased=False)

    ax_orbit.set_aspect("equal", adjustable="box")
    ax_orbit.set_xlabel("x [m]")
    ax_orbit.set_ylabel("y [m]")
    ax_orbit.set_title(f"Orbit animation {ecc_text} (time in years)")
    ax_orbit.grid(True, ls="--", alpha=0.35)
    ax_orbit.legend(loc="upper right")

    # Orbit bounds
    x_all = np.concatenate([xN, xG])
    y_all = np.concatenate([yN, yG])
    dx = (np.max(x_all) - np.min(x_all)) * 0.05 + 1e-9
    dy = (np.max(y_all) - np.min(y_all)) * 0.05 + 1e-9
    ax_orbit.set_xlim(np.min(x_all) - dx, np.max(x_all) + dx)
    ax_orbit.set_ylim(np.min(y_all) - dy, np.max(y_all) + dy)

    # METRICS: eccentricity and 1PN energy (relative)
    ax_metrics.plot(t_yr, eN, label="eN", color="#1f77b4", lw=1.2, antialiased=False)
    ax_metrics.plot(t_yr, eG, label="eGR", color="#d62728", lw=1.2, antialiased=False)
    ax_metrics.plot(t_yr, dEN_rel,   label="(EN-EN0)/|EN0|",   color="#2ca02c", lw=1.0, alpha=0.9, antialiased=False)
    ax_metrics.plot(t_yr, dE1PN_rel, label="(E1PN_GR-E1PN0)/|E1PN0|", color="#ff7f0e", lw=1.0, alpha=0.9, antialiased=False)

    t_cursor = ax_metrics.axvline(t_yr[0], color="k", lw=1.0, alpha=0.7)
    time_text = ax_metrics.text(0.02, 0.95, "", transform=ax_metrics.transAxes, va="top")

    ax_metrics.set_xlabel("t [years]")
    ax_metrics.set_ylabel("Eccentricity / Relative energy change")
    ax_metrics.grid(True, ls="--", alpha=0.35)
    ax_metrics.legend(loc="best")

    plt.tight_layout()

    frames = len(df)
    interval_ms = 1000.0 / max(1, args.fps)
    trail_len = max(1, args.trail)

    def init():
        headN.set_data([], [])
        headG.set_data([], [])
        trailN.set_data([], [])
        trailG.set_data([], [])
        t_cursor.set_xdata([t_yr[0], t_yr[0]])
        time_text.set_text(f"t = {t_yr[0]:.3f} yr")
        return headN, headG, trailN, trailG, t_cursor, time_text

    def update(i):
        i0 = max(0, i - trail_len)
        headN.set_data([xN[i]], [yN[i]])
        headG.set_data([xG[i]], [yG[i]])
        trailN.set_data(xN[i0:i+1], yN[i0:i+1])
        trailG.set_data(xG[i0:i+1], yG[i0:i+1])
        t_cursor.set_xdata([t_yr[i], t_yr[i]])
        time_text.set_text(f"t = {t_yr[i]:.3f} yr")
        return headN, headG, trailN, trailG, t_cursor, time_text

    print(f"[4/6] Creating animation object (frames={frames}, fps={args.fps})...", flush=True)
    anim = FuncAnimation(fig, update, init_func=init, frames=frames, interval=interval_ms, blit=True)

    if args.out:
        print(f"[5/6] Saving video to {args.out} ...", flush=True)

        def progress_callback(cur, total):
            pct = int(100 * (cur + 1) / total)
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(f"  [{bar}] {pct:3d}% ({cur+1}/{total})", end="\r", flush=True)
            if cur + 1 == total:
                print()

        try:
            if args.out.lower().endswith(".mp4"):
                anim.save(
                    args.out, fps=args.fps, dpi=args.save_dpi, writer="ffmpeg",
                    savefig_kwargs={"facecolor": "white"}, progress_callback=progress_callback
                )
            elif args.out.lower().endswith(".gif"):
                anim.save(
                    args.out, fps=args.fps, dpi=max(80, args.save_dpi), writer="pillow",
                    savefig_kwargs={"facecolor": "white"}, progress_callback=progress_callback
                )
            else:
                print("  Unknown extension; skipping save. Use .mp4 or .gif.")
        except Exception as ex:
            print(f"  Save failed: {ex}")

    print(f"[6/6] Displaying animation for {args.display_seconds}s, then exiting...", flush=True)
    plt.show(block=False)
    try:
        mng = plt.get_current_fig_manager()
        mng.window.wm_geometry("+120+60")
    except Exception:
        pass

    for sec in range(args.display_seconds, 0, -1):
        print(f"  Closing in {sec:2d}s", end="\r", flush=True)
        time.sleep(1)
    print("  Closing now  ", flush=True)

    plt.close(fig)
    plt.close("all")
    sys.exit(0)

if __name__ == "__main__":
    main()
