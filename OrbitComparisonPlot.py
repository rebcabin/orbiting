#! python
import pandas as pd
import matplotlib.pyplot as plt

SECONDS_PER_YEAR = 365.25 * 24 * 3600.0

def main():
    df = pd.read_csv("orbits.csv")

    # Verify required columns (now requires E1PN_GR)
    required = {"t", "xN", "yN", "xGR", "yGR", "eN", "eGR", "EN", "EGR", "E1PN_GR"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing columns in CSV: {missing}")
        return

    # Labels
    eN0 = float(df.iloc[0]["eN"])
    eGR0 = float(df.iloc[0]["eGR"])
    ecc_text = f"(eN≈{eN0:.3f}, eGR≈{eGR0:.3f})"

    # Time (use years on plots)
    t_sec = df["t"].values
    t_yr = t_sec / SECONDS_PER_YEAR

    # Relative energy changes
    EN0 = float(df.iloc[0]["EN"])
    dEN_rel = (df["EN"].values - EN0) / abs(EN0)

    # Use proper 1PN energy for GR curve
    E1PN0 = float(df.iloc[0]["E1PN_GR"])
    dE1PN_rel = (df["E1PN_GR"].values - E1PN0) / abs(E1PN0)

    # Figure with three subplots: orbit, eccentricity, energy
    fig, (ax_orbit, ax_e, ax_E) = plt.subplots(
        3, 1, figsize=(9, 12), gridspec_kw={"height_ratios": [3, 2, 2]}
    )

    # Top: orbit plot
    ax_orbit.plot(df["xN"], df["yN"], label="Newtonian", lw=1.2)
    ax_orbit.plot(df["xGR"], df["yGR"], label="GR (1PN)", lw=1.2)
    ax_orbit.scatter(0.0, 0.0, s=60, color="black", marker="o", label="SMBH", zorder=5)
    ax_orbit.set_aspect("equal", adjustable="box")
    ax_orbit.set_xlabel("x [m]")
    ax_orbit.set_ylabel("y [m]")
    ax_orbit.set_title(f"Orbit comparison: Newtonian vs GR (1PN) {ecc_text}")
    ax_orbit.grid(True, ls="--", alpha=0.4)
    ax_orbit.legend()

    # Middle: eccentricity vs time (years)
    ax_e.plot(t_yr, df["eN"].values, label="eN (Newtonian)", lw=1.4)
    ax_e.plot(t_yr, df["eGR"].values, label="eGR (GR 1PN)", lw=1.4)
    ax_e.set_xlabel("t [years]")
    ax_e.set_ylabel("Eccentricity")
    ax_e.set_title("Eccentricity evolution")
    ax_e.grid(True, ls="--", alpha=0.4)
    ax_e.legend()

    # Bottom: relative energy change vs time (use EN for Newtonian, E1PN_GR for GR)
    ax_E.plot(t_yr, dEN_rel, label="(EN - EN0)/|EN0| (Newtonian)", lw=1.4)
    ax_E.plot(t_yr, dE1PN_rel, label="(E1PN_GR - E1PN0)/|E1PN0| (GR 1PN)", lw=1.4)
    ax_E.set_xlabel("t [years]")
    ax_E.set_ylabel("Relative energy change")
    ax_E.set_title("Energy stability (relative change)")
    ax_E.grid(True, ls="--", alpha=0.4)
    ax_E.legend(loc="best")

    # Optional: secondary x-axis in years for the bottom two subplots
    try:
        from matplotlib.ticker import FuncFormatter

        secax_e = ax_e.secondary_xaxis(
            "top",
            functions=(lambda s: s / (365.25 * 24 * 3600.0),
                       lambda yr: yr * (365.25 * 24 * 3600.0)),
        )
        secax_e.set_xlabel("t [years]")
        secax_e.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))

        secax_E = ax_E.secondary_xaxis(
            "top",
            functions=(lambda s: s / (365.25 * 24 * 3600.0),
                       lambda yr: yr * (365.25 * 24 * 3600.0)),
        )
        secax_E.set_xlabel("t [years]")
        secax_E.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    except Exception:
        pass

    plt.tight_layout()
    plt.savefig("orbit_ecc_energy.png", dpi=200)

    # Show for 15 seconds, then close and exit
    fig.set_size_inches(*(fig.get_size_inches() * 0.8), forward=True)
    plt.tight_layout(pad=1.2)
    plt.show(block=False)
    try:
        mng = plt.get_current_fig_manager()
        # geometry: "WIDTHxHEIGHT+X+Y" in pixels; adjust Y (last number) smaller to move up
        mng.window.wm_geometry("+100+50")
    except Exception:
        pass
    plt.pause(15.0)
    plt.close(fig)

if __name__ == "__main__":
    main()
