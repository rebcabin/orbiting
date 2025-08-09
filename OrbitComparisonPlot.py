#! python
import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("orbits.csv")

    # Use the first sample as representative eccentricities
    try:
        eN0 = float(df.loc[0, "eN"])
        eGR0 = float(df.loc[0, "eGR"])
        ecc_text = f" (eN≈{eN0:.3f}, eGR≈{eGR0:.3f})"
    except Exception:
        ecc_text = ""

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(df["xN"], df["yN"], label="Newtonian", lw=1.2)
    ax.plot(df["xGR"], df["yGR"], label="GR (1PN)", lw=1.2)

    # Supermassive black hole at the origin
    ax.scatter(0.0, 0.0, s=60, color="black", marker="o", label="SMBH", zorder=5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Orbit comparison: Newtonian vs GR (1PN)" + ecc_text)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()

    plt.tight_layout()
    plt.savefig("orbit.png", dpi=200)

    # Show for 15 seconds, then close and exit
    plt.show(block=False)
    plt.pause(15.0)
    plt.close(fig)

if __name__ == "__main__":
    main()
