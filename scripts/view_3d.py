"""Interactive 3D viewer of the SEM twin.

    python scripts/view_3d.py
    python scripts/view_3d.py --planes -80 -40 0 40 80 --window 60

Left: the column in 3D with traced rays (drag to rotate). Middle: close-up of the focus with the
PSF slices drawn at their heights. Right: the PSF in each slice. Move the sliders to change the
machine; the view redraws shortly after you let go.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semtwin.column import SEMColumn, SEMSettings
from semtwin.plotting import draw_overview, settings_label

# knob, min, max, label
KNOBS = [
    ("defocus_um", -100, 100, "stage defocus [um]"),
    ("obj_strength", 0.995, 1.005, "objective strength"),
    ("aperture_radius_mm", 0.2, 2.0, "aperture radius [mm]"),
    ("aperture_shift_x_mm", -0.5, 0.5, "aperture shift x [mm]"),
    ("aperture_shift_y_mm", -0.5, 0.5, "aperture shift y [mm]"),
    ("stig_x", -6e-4, 6e-4, "stigmator x [1/mm]"),
    ("stig_y", -6e-4, 6e-4, "stigmator y [1/mm]"),
    ("obj_tilt_x_mrad", -60, 60, "objective tilt x [mrad]"),
    ("obj_tilt_y_mrad", -60, 60, "objective tilt y [mrad]"),
    ("obj_shift_x_mm", -0.4, 0.4, "objective shift x [mm]"),
    ("c2_strength", 0.97, 1.03, "condenser 2 strength"),
    ("bandwidth_nm", 0, 40, "bandwidth [nm]"),
    ("source_radius_um", 0, 10, "source radius [um]"),
    ("scan_x_mm", -1.5, 1.5, "scan x [mm]"),
    ("scan_y_mm", -1.5, 1.5, "scan y [mm]"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--planes", type=float, nargs="+", default=[-50.0, 0.0, 50.0],
                   help="PSF slice offsets from the specimen [um]")
    p.add_argument("--window", type=float, default=None, help="fixed PSF window [um] (default: automatic)")
    p.add_argument("--rays", type=int, default=2 ** 16, help="rays per trace")
    args = p.parse_args()

    column = SEMColumn(n_rays=args.rays)
    default = SEMSettings()
    fig = plt.figure(figsize=(17, 10.5))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[3.2, 1], left=0.01, right=0.98, top=0.95,
                     bottom=0.02, hspace=0.12)
    state = dict(axes=[], views=None)

    # sliders in three columns
    sliders = {}
    n_per_col = 5
    for k, (name, lo, hi, label) in enumerate(KNOBS):
        col, row = divmod(k, n_per_col)
        ax = fig.add_axes([0.08 + col * 0.31, 0.205 - row * 0.038, 0.17, 0.022])
        sliders[name] = Slider(ax, label, lo, hi, valinit=getattr(default, name),
                               valfmt="%.4g" if name.startswith("stig") or "strength" in name else "%.3g")
        sliders[name].label.set_fontsize(8)
    ax_reset = fig.add_axes([0.90, 0.17, 0.07, 0.035])
    ax_save = fig.add_axes([0.90, 0.12, 0.07, 0.035])
    b_reset, b_save = Button(ax_reset, "Reset"), Button(ax_save, "Save PNG")

    def current_settings():
        return default.replace(**{k: float(sl.val) for k, sl in sliders.items()})

    def redraw():
        views = [(a.elev, a.azim) for a in state["axes"][:2]] if state["axes"] else None
        for a in state["axes"]:
            a.remove()
        s = current_settings()
        state["axes"], _ = draw_overview(fig, outer[0], column, s, args.planes, args.window)
        if views:
            for a, (el, az) in zip(state["axes"][:2], views):
                a.view_init(el, az)
        fig.suptitle(settings_label(s), fontsize=10)
        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=350)
    timer.single_shot = True
    timer.add_callback(redraw)

    def schedule(_=None):
        timer.stop()
        timer.start()

    for sl in sliders.values():
        sl.on_changed(schedule)

    def reset(_):
        for sl in sliders.values():
            sl.eventson = False
            sl.reset()
            sl.eventson = True
        redraw()

    def save(_):
        out = Path(__file__).resolve().parents[1] / "output" / "figures"
        out.mkdir(parents=True, exist_ok=True)
        k = len(list(out.glob("viewer_*.png")))
        fig.savefig(out / f"viewer_{k:03d}.png", dpi=130)
        print("saved", out / f"viewer_{k:03d}.png")

    b_reset.on_clicked(reset)
    b_save.on_clicked(save)
    redraw()
    plt.show()


if __name__ == "__main__":
    main()
