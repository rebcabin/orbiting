#! python
import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("orbits.csv")

    # Verify required columns
    required = {"t", "xN", "yN", "xGR", "yGR", "eN", "eGR", "EN", "EGR"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing columns in CSV: {missing}")
        return

    # Labels
    eN0 = float(df.iloc[0]["eN"])
    eGR0 = float(df.iloc[0]["eGR"])
    ecc_text = f"(eN≈{eN0:.3f}, eGR≈{eGR0:.3f})"

    # Time
    t = df["t"].values

    # Relative energy changes
    EN0 = float(df.iloc[0]["EN"])
    EGR0 = float(df.iloc[0]["EGR"])
    dEN_rel = (df["EN"].values - EN0) / abs(EN0)
    dEGR_rel = (df["EGR"].values - EGR0) / abs(EGR0)

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

    # Middle: eccentricity vs time
    ax_e.plot(t, df["eN"].values, label="eN (Newtonian)", lw=1.4)
    ax_e.plot(t, df["eGR"].values, label="eGR (GR 1PN)", lw=1.4)
    ax_e.set_xlabel("t [s]")
    ax_e.set_ylabel("Eccentricity")
    ax_e.set_title("Eccentricity evolution")
    ax_e.grid(True, ls="--", alpha=0.4)
    ax_e.legend()

    # Bottom: relative energy change vs time
    ax_E.plot(t, dEN_rel, label="(EN - EN0)/|EN0| (Newtonian)", lw=1.4)
    ax_E.plot(t, dEGR_rel, label="(EGR - EGR0)/|EGR0| (GR 1PN)", lw=1.4)
    ax_E.set_xlabel("t [s]")
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
    plt.show(block=False)
    plt.pause(15.0)
    plt.close(fig)

if __name__ == "__main__":
    main()
