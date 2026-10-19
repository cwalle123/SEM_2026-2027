"""Generate all overview figures into output/figures.

    python scripts/make_figures.py            # everything
    python scripts/make_figures.py --only column sweeps
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semtwin.column import SEMColumn, SEMSettings
from semtwin.database import load_database
from semtwin.plotting import (column_figure, sweep_figure, through_focus_figure, field_map_figure,
                              sample_images_figure, database_examples_figure, database_sensitivity_figure,
                              settings_label)
from semtwin.samples import make_samples

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "figures"

N = SEMSettings()
CONFIGS = [
    ("nominal", N),
    ("defocus +40 um", N.replace(defocus_um=40)),
    ("large aperture (spherical)", N.replace(aperture_radius_mm=1.6)),
    ("stigmator x (astigmatism)", N.replace(stig_x=3e-4)),
    ("stigmator y (astigmatism)", N.replace(stig_y=3e-4)),
    ("objective tilt 40 mrad", N.replace(obj_tilt_x_mrad=40, aperture_radius_mm=1.3)),
    ("aperture off-centre 0.4 mm", N.replace(aperture_shift_x_mm=0.4)),
    ("bandwidth 25 nm (chromatic)", N.replace(bandwidth_nm=25)),
    ("off-axis scan (0.8, 0.5) mm", N.replace(scan_x_mm=0.8, scan_y_mm=0.5)),
]

SWEEPS = [
    ("defocus_um", [-60, -30, 0, 30, 60], "stage defocus [um]"),
    ("obj_strength", [0.997, 0.9985, 1.0, 1.0015, 1.003], "objective strength"),
    ("aperture_radius_mm", [0.4, 0.8, 1.2, 1.6, 2.0], "aperture radius [mm]"),
    ("aperture_shift_x_mm", [-0.5, -0.25, 0, 0.25, 0.5], "aperture shift x [mm]"),
    ("stig_x", [-4e-4, -2e-4, 0, 2e-4, 4e-4], "stigmator x [1/mm]"),
    ("stig_y", [-4e-4, -2e-4, 0, 2e-4, 4e-4], "stigmator y [1/mm]"),
    ("obj_tilt_x_mrad", [-60, -30, 0, 30, 60], "objective tilt x [mrad]"),
    ("obj_shift_x_mm", [-0.4, -0.2, 0, 0.2, 0.4], "objective shift x [mm]"),
    ("c2_strength", [0.98, 0.99, 1.0, 1.01, 1.02], "condenser 2 strength"),
    ("bandwidth_nm", [0, 10, 20, 30, 40], "bandwidth [nm]"),
    ("source_radius_um", [0, 3, 6, 9, 12], "source radius [um]"),
    ("scan_x_mm", [-1.5, -0.75, 0, 0.75, 1.5], "scan position x [mm]"),
]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=130)
    plt.close(fig)
    print("  wrote", OUT / name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=None,
                   help="subset of: column sweeps focus field samples database")
    args = p.parse_args()
    want = lambda k: args.only is None or k in args.only
    col = SEMColumn()
    print(f"specimen (best focus) at z = {col.specimen_z:.3f} mm")

    if want("column"):
        for i, (name, s) in enumerate([CONFIGS[0], CONFIGS[3], CONFIGS[5], CONFIGS[7]]):
            fig, _ = column_figure(col, s, title=f"{name}  ({settings_label(s)})")
            save(fig, f"01_column_{i}_{name.split(' (')[0].replace(' ', '_')}.png")
    if want("sweeps"):
        save(sweep_figure(col, SWEEPS), "02_knob_sweeps.png")
    if want("focus"):
        save(through_focus_figure(col, CONFIGS), "03_through_focus.png")
    if want("field"):
        save(field_map_figure(col, [("nominal", N), ("objective tilt 40 mrad", CONFIGS[5][1])]), "04_field_map.png")
    if want("samples"):
        samples = make_samples(160, seed=1)
        save(sample_images_figure(col, [CONFIGS[i] for i in (0, 1, 3, 5, 6, 7)], samples, fov_mm=0.3),
             "05_sample_images.png")
    if want("database"):
        path = ROOT / "output" / "database" / "sem_ray_db.npz"
        if path.exists():
            db = load_database(path)
            save(database_examples_figure(db), "06_database_examples.png")
            save(database_sensitivity_figure(db), "07_database_sensitivity.png")
        else:
            print("  (no database yet: run scripts/build_database.py first)")


if __name__ == "__main__":
    main()
