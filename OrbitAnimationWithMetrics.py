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
    parser = argparse.ArgumentParser(description="Animate orbit and element evolution with progress and year timebase.")
    parser.add_argument("--csv", default="orbits.csv", help="Path to CSV file")
    parser.add_argument("--out", default="orbit_animation.mp4", help="Output video (mp4 or gif). Leave as .mp4 or .gif, or set to '' to skip saving.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--trail", type=int, default=400, help="Trail length in samples")
    parser.add_argument("--step", type=int, default=5, help="Use every Nth row to speed up")
    parser.add_argument("--duration", type=float, default=None, help="Limit animation to first D seconds (simulation time)")
    parser.add_argument("--max-frames", type=int, default=2500, help="Cap total frames by uniform subsampling")
    parser.add_argument("--save-dpi", type=int, default=120, help="DPI when saving video")
    parser.add_argument("--display-seconds", type=int, default=12, help="Seconds to keep the window open before exit")
    args = parser.parse_args()

    print("[1/6] Reading CSV...", flush=True)
    df = pd.read_csv(args.csv)

    print("[2/6] Preparing data...", flush=True)
    # Optional time limit (in seconds)
    if args.duration is not None:
        df = df[df["t"] <= args.duration].reset_index(drop=True)

    # Base decimation
    df = df.iloc[::max(1, args.step)].reset_index(drop=True)

    # Cap total frames by uniform subsampling
    if args.max_frames is not None and len(df) > args.max_frames:
        idx = np.linspace(0, len(df) - 1, args.max_frames).astype(int)
        df = df.iloc[idx].reset_index(drop=True)

    # Verify columns
    required = {"t","xN","yN","xGR","yGR","eN","eGR","EN","EGR"}
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
    EN, EG = df["EN"].to_numpy(), df["EGR"].to_numpy()

    # Relative energies
    EN0, EG0 = EN[0], EG[0]
    dEN_rel = (EN - EN0) / abs(EN0)
    dEG_rel = (EG - EG0) / abs(EG0)

    # Labels
    eN0, eG0 = float(eN[0]), float(eG[0])
    ecc_text = f"(eN≈{eN0:.3f}, eGR≈{eG0:.3f})"

    print("[3/6] Building figure and static elements...", flush=True)
    # Figure slightly smaller to fit screens better
    fig, (ax_orbit, ax_metrics) = plt.subplots(2, 1, figsize=(8, 9), gridspec_kw={"height_ratios": [3, 2]})
    fig.set_size_inches(*(fig.get_size_inches() * 0.85), forward=True)

    # Orbit panel: full paths
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

    # Autoscale orbit view with padding
    x_all = np.concatenate([xN, xG])
    y_all = np.concatenate([yN, yG])
    dx = (np.max(x_all) - np.min(x_all)) * 0.05 + 1e-9
    dy = (np.max(y_all) - np.min(y_all)) * 0.05 + 1e-9
    ax_orbit.set_xlim(np.min(x_all) - dx, np.max(x_all) + dx)
    ax_orbit.set_ylim(np.min(y_all) - dy, np.max(y_all) + dy)

    # Metrics: eccentricity and relative energy vs time (in years)
    ax_metrics.plot(t_yr, eN, label="eN", color="#1f77b4", lw=1.2, antialiased=False)
    ax_metrics.plot(t_yr, eG, label="eGR", color="#d62728", lw=1.2, antialiased=False)
    ax_metrics.plot(t_yr, dEN_rel, label="(EN-EN0)/|EN0|", color="#2ca02c", lw=1.0, alpha=0.9, antialiased=False)
    ax_metrics.plot(t_yr, dEG_rel, label="(EGR-EGR0)/|EGR0|", color="#ff7f0e", lw=1.0, alpha=0.9, antialiased=False)

    # Moving time cursor in years
    t_cursor = ax_metrics.axvline(t_yr[0], color="k", lw=1.0, alpha=0.7)
    time_text = ax_metrics.text(0.02, 0.95, "", transform=ax_metrics.transAxes, va="top")

    ax_metrics.set_xlabel("t [years]")
    ax_metrics.set_ylabel("Eccentricity / Relative energy change")
    ax_metrics.grid(True, ls="--", alpha=0.35)
    ax_metrics.legend(loc="best")

    plt.tight_layout()

    # Animation setup
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
        # heads
        headN.set_data([xN[i]], [yN[i]])
        headG.set_data([xG[i]], [yG[i]])
        # trails
        trailN.set_data(xN[i0:i+1], yN[i0:i+1])
        trailG.set_data(xG[i0:i+1], yG[i0:i+1])
        # cursor and text (years)
        t_cursor.set_xdata([t_yr[i], t_yr[i]])
        time_text.set_text(f"t = {t_yr[i]:.3f} yr")
        return headN, headG, trailN, trailG, t_cursor, time_text

    print(f"[4/6] Creating animation object (frames={frames}, fps={args.fps})...", flush=True)
    anim = FuncAnimation(fig, update, init_func=init, frames=frames, interval=interval_ms, blit=True)

    # Save video if requested
    if args.out:
        print(f"[5/6] Saving video to {args.out} ...", flush=True)

        # live progress callback (Matplotlib >= 3.4)
        def progress_callback(cur, total):
            pct = int(100 * (cur + 1) / total)
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(f"  [{bar}] {pct:3d}% ({cur+1}/{total})", end="\r", flush=True)
            if cur + 1 == total:
                print()  # newline at end

        try:
            if args.out.lower().endswith(".mp4"):
                anim.save(
                    args.out,
                    fps=args.fps,
                    dpi=args.save_dpi,
                    writer="ffmpeg",
                    savefig_kwargs={"facecolor": "white"},
                    progress_callback=progress_callback,
                )
            elif args.out.lower().endswith(".gif"):
                anim.save(
                    args.out,
                    fps=args.fps,
                    dpi=max(80, args.save_dpi),
                    writer="pillow",
                    savefig_kwargs={"facecolor": "white"},
                    progress_callback=progress_callback,
                )
            else:
                print("  Unknown extension; skipping save. Use .mp4 (ffmpeg) or .gif (pillow).")
        except Exception as ex:
            print(f"  Save failed: {ex}")

    # Show the window briefly, then exit
    print(f"[6/6] Displaying animation for {args.display_seconds}s, then exiting...", flush=True)
    plt.show(block=False)
    try:
        mng = plt.get_current_fig_manager()
        # geometry: "+X+Y" in pixels; smaller Y moves window up
        mng.window.wm_geometry("+120+60")
    except Exception:
        pass

    # Countdown in console
    for sec in range(args.display_seconds, 0, -1):
        print(f"  Closing in {sec:2d}s", end="\r", flush=True)
        time.sleep(1)
    print("  Closing now  ", flush=True)

    plt.close(fig)
    plt.close("all")
    sys.exit(0)

if __name__ == "__main__":
    main()
