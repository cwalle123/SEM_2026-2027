"""Visualisation: the 3D column with rays, the 3D focus region with PSF slices, and 2D PSF grids."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from .optics import Lens, Aperture, Deflector, Stigmator
from .psf import probe_psfs, spot_metrics

PSF_CMAP = "inferno"
GLASS = (0.55, 0.75, 0.95)


# ------------------------------------------------------------------ 3D primitives
def _cylinder(ax, r, z0, z1, centre=(0, 0), color="0.5", alpha=0.3, n=40):
    phi, z = np.meshgrid(np.linspace(0, 2 * np.pi, n), [z0, z1])
    ax.plot_surface(centre[0] + r * np.cos(phi), centre[1] + r * np.sin(phi), z,
                    color=color, alpha=alpha, linewidth=0, shade=False)


def _annulus(ax, r0, r1, z, centre=(0, 0), color="0.3", alpha=0.8, n=48):
    r, phi = np.meshgrid([r0, r1], np.linspace(0, 2 * np.pi, n))
    ax.plot_surface(centre[0] + r * np.cos(phi), centre[1] + r * np.sin(phi), np.full_like(r, z),
                    color=color, alpha=alpha, linewidth=0, shade=False)


def _draw_lens(ax, lens: Lens, n_r=10, n_phi=40):
    r, phi = np.meshgrid(np.linspace(0, lens.semi_diameter, n_r), np.linspace(0, 2 * np.pi, n_phi))
    R, o = lens.rotation, lens.origin
    h = lens.thickness / 2
    rims = []
    for z_v, c in zip((-h, h), lens.curvatures):
        sag = c * r ** 2 / (1 + np.sqrt(np.maximum(1 - (c * r) ** 2, 0)))
        local = np.stack([r * np.cos(phi), r * np.sin(phi), z_v + sag], -1)
        g = local @ R.T + o
        ax.plot_surface(g[..., 0], g[..., 1], g[..., 2], color=GLASS, alpha=0.35, linewidth=0, shade=True)
        rims.append(g[:, -1])
    edge = np.stack(rims, 1)  # (n_phi, 2, 3)
    ax.plot_surface(edge[..., 0], edge[..., 1], edge[..., 2], color=GLASS, alpha=0.5, linewidth=0, shade=True)


def _ray_paths(rays):
    """Per ray: the polyline of positions while it was alive (plus the point where it died)."""
    P = np.stack(rays.path, 1)            # (N, steps, 3)
    A = np.stack(rays.path_alive, 1)      # alive after each step
    keep = np.concatenate([np.ones((len(P), 1), bool), A[:, :-1]], 1)
    return [P[i][keep[i]] for i in range(len(P))], A[:, -1]


def wavelength_color(wl_nm):
    """Rough visible-spectrum colour for a wavelength."""
    t = np.clip((wl_nm - 440) / (660 - 440), 0, 1)
    return cm.turbo(0.1 + 0.8 * t)


# ------------------------------------------------------------------ the column
def draw_column(ax, column, s, slice_offsets_um=(0.0,), labels=True):
    """3D drawing of the whole column with a fan of traced rays and the PSF slice plane(s)."""
    g = column.geometry
    elements = column.elements(s)
    z_spec = column.specimen_plane(s)

    # gun: a cone ending in the emitting tip
    phi, zz = np.meshgrid(np.linspace(0, 2 * np.pi, 30), [g.source_z - 12, g.source_z])
    rr = (g.source_z - zz) / 12 * 2.5
    ax.plot_surface(rr * np.cos(phi), rr * np.sin(phi), zz, color="0.6", alpha=0.6, linewidth=0, shade=True)
    ax.scatter([0], [0], [g.source_z], color="gold", s=30, depthshade=False)

    for el in elements:
        if isinstance(el, Lens):
            _draw_lens(ax, el)
        elif isinstance(el, Aperture):
            _annulus(ax, el.radius, el.outer_radius, el.z, el.shift, "0.25", 0.75)
        elif isinstance(el, Deflector):
            _cylinder(ax, 4.5, el.z - 3, el.z + 3, color=(0.80, 0.45, 0.15), alpha=0.25)
        elif isinstance(el, Stigmator):
            _cylinder(ax, 4.0, el.z - 1.5, el.z + 1.5, color=(0.55, 0.3, 0.75), alpha=0.25)
            for k in range(4):
                a = np.pi / 4 + k * np.pi / 2
                ax.plot([3.0 * np.cos(a), 4.0 * np.cos(a)], [3.0 * np.sin(a), 4.0 * np.sin(a)],
                        [el.z, el.z], color=(0.55, 0.3, 0.75), lw=4)
        if labels:
            ax.text(7.2, 0, el.z, el.name, fontsize=8, va="center")
    if labels:
        ax.text(7.2, 0, g.source_z - 6, "Gun (light source)", fontsize=8, va="center")

    # specimen stage
    sq = np.array([-6, 6])
    X, Y = np.meshgrid(sq, sq)
    ax.plot_surface(X, Y, np.full_like(X, z_spec, dtype=float), color="0.55", alpha=0.35, linewidth=0, shade=False)
    if labels:
        ax.text(7.2, 0, z_spec + 4, "Specimen", fontsize=8, va="center")

    # rays
    rays, ring = column.demo_rays(s)
    paths, alive = _ray_paths(rays)
    z_end = z_spec + max(max(slice_offsets_um), 0) * 1e-3
    xe, ye = rays.at_z(z_end)
    end = np.full((len(rays), 3), np.nan)
    end[alive] = np.stack([xe, ye, np.full_like(xe, z_end)], 1)
    for i, p in enumerate(paths):
        if alive[i]:
            p = np.vstack([p, end[i]])
            col = wavelength_color(rays.wl[i]) if s.bandwidth_nm > 0 else cm.viridis(0.15 + 0.75 * ring[i])
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color=col, lw=0.7, alpha=0.9)
        else:
            ax.plot(p[:, 0], p[:, 1], p[:, 2], color="0.6", lw=0.6, ls="--", alpha=0.7)

    # where the PSF slices are taken (they are microns wide: see the zoom panel)
    t = np.linspace(0, 2 * np.pi, 60)
    for off in slice_offsets_um:
        z = column.specimen_plane(s, off)
        ax.plot(s.scan_x_mm + 1.2 * np.cos(t), s.scan_y_mm + 1.2 * np.sin(t), np.full_like(t, z),
                color="red", lw=1.2)
    if labels:
        ax.text(-7, 0, z_spec, "PSF slice", color="red", fontsize=8, ha="right")

    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_zlim(z_spec + 10, g.source_z - 14)  # z downwards: gun on top
    ax.set_box_aspect((1, 1, 2.6))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.set_title("SEM column (radial scale exaggerated)", fontsize=10)


def _clip_segment(x0, y0, u, v, half, z0, z1):
    """z-interval in [z0, z1] where the line x0 + u z, y0 + v z stays inside |x|,|y| <= half."""
    lo, hi = z0, z1
    for a, b in ((x0, u), (y0, v)):
        if abs(b) < 1e-15:
            if abs(a) > half:
                return None
            continue
        za, zb = sorted(((-half - a) / b, (half - a) / b))
        lo, hi = max(lo, za), min(hi, zb)
    return (lo, hi) if hi > lo else None


def draw_focus_zoom(ax, column, s, probe, psfs, centre, window_um, n_rays=250, texture_px=48):
    """3D close-up of the beam around the specimen: rays converging plus the PSF slices drawn as
    textured planes at their heights. Coordinates in um relative to the probe centre / specimen."""
    half = window_um / 2
    offs = probe.planes_um
    zr = 1.35 * max(np.abs(offs).max(), 20.0)
    z_spec = column.specimen_plane(s)

    rays = probe.rays
    idx = np.flatnonzero(rays.alive)
    idx = idx[np.linspace(0, len(idx) - 1, min(n_rays, len(idx))).astype(int)]
    p, d = rays.pos[idx], rays.dir[idx]
    u, v = d[:, 0] / d[:, 2], d[:, 1] / d[:, 2]
    # line through the specimen plane, in um relative to centre
    x0 = (p[:, 0] + u * (z_spec - p[:, 2]) - centre[0]) * 1e3
    y0 = (p[:, 1] + v * (z_spec - p[:, 2]) - centre[1]) * 1e3
    cols = wavelength_color(rays.wl[idx]) if s.bandwidth_nm > 0 else None
    for k in range(len(idx)):
        seg = _clip_segment(x0[k], y0[k], u[k], v[k], half, -zr, zr)
        if seg is None:
            continue
        z = np.array(seg)
        ax.plot(x0[k] + u[k] * z, y0[k] + v[k] * z, z, lw=0.4, alpha=0.35,
                color=cols[k] if cols is not None else (0.2, 0.6, 0.9))

    # textured PSF slices
    n = psfs.shape[-1]
    step = max(n // texture_px, 1)
    edges = np.linspace(-half, half, n // step + 1)
    X, Y = np.meshgrid(edges, edges)
    for img, off in zip(psfs, offs):
        small = img[: (n // step) * step, : (n // step) * step].reshape(n // step, step, n // step, step).mean((1, 3))
        val = small / small.max() if small.max() > 0 else small
        rgba = plt.get_cmap(PSF_CMAP)(0.25 + 0.75 * np.sqrt(val))
        rgba[..., 3] = np.clip(1.5 * np.sqrt(val), 0, 1)  # empty pixels fully transparent: no seams
        ax.plot_surface(X, Y, np.full_like(X, off), facecolors=rgba, rstride=1, cstride=1,
                        linewidth=0, edgecolor="none", shade=False, antialiased=False)
        corners = np.array([-half, half])
        ax.plot_surface(*np.meshgrid(corners, corners), np.full((2, 2), off), color="0.5", alpha=0.08,
                        linewidth=0, shade=False)
        ax.plot([-half, half, half, -half, -half], [-half, -half, half, half, -half], [off] * 5,
                color="red" if off == 0 else "0.4", lw=0.8)
        name = "specimen plane" if off == 0 else (f"{off:+.0f} um ({'under' if off < 0 else 'over'} focus)")
        ax.text(half, half, off, "  " + name, fontsize=8)

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_zlim(zr, -zr)
    ax.set_box_aspect((1, 1, 1.6))
    ax.set_xlabel("x [um]")
    ax.set_ylabel("y [um]")
    ax.set_zlabel("z from specimen [um]")
    ax.set_title("Focus region: rays and PSF slices", fontsize=10)


def show_psf(ax, img, window_um, title=None, colorbar=False, normalise=True, gamma=0.5):
    """Show a PSF (or image) with a physical extent in um. gamma < 1 brightens faint structure
    (ray caustics are very peaked); use gamma=1 for a linear scale."""
    half = window_um / 2
    v = img / img.max() if normalise and img.max() > 0 else img
    im = ax.imshow(v, origin="lower", extent=(-half, half, -half, half), cmap=PSF_CMAP,
                   interpolation="nearest", norm=PowerNorm(gamma, vmin=0, vmax=v.max() or 1))
    if title:
        ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    if colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046)
    return im


def auto_window(probe, minimum=16.0, maximum=200.0):
    """A PSF window [um] that holds ~all of the rays on every plane."""
    x, y = probe.hits[probe.plane_index(0.0)]
    c = np.array([x.mean(), y.mean()])
    r = max(np.percentile(np.hypot(xx - c[0], yy - c[1]), 99.5) for xx, yy in probe.hits if len(xx)) * 1e3
    return float(np.clip(2.3 * r, minimum, maximum))


def draw_overview(fig, area, column, s, planes_um=(-50.0, 0.0, 50.0), window_um=None, npix=96):
    """3D column | 3D focus close-up with slices | 2D PSF per slice, drawn into the SubplotSpec
    `area` of `fig`. Returns the created axes and the computed probe/PSFs."""
    probe = column.probe(s, planes_um)
    window_um = window_um or auto_window(probe)
    psfs, centre = probe_psfs(probe, window_um, npix)
    n = len(planes_um)
    gs = GridSpecFromSubplotSpec(n, 3, subplot_spec=area, width_ratios=[0.75, 1.25, 0.45],
                                 wspace=0.02, hspace=0.45)
    ax_col = fig.add_subplot(gs[:, 0], projection="3d")
    ax_zoom = fig.add_subplot(gs[:, 1], projection="3d")
    draw_column(ax_col, column, s, planes_um)
    draw_focus_zoom(ax_zoom, column, s, probe, psfs, centre, window_um)
    axes = [ax_col, ax_zoom]
    for i, (img, off) in enumerate(zip(psfs, probe.planes_um)):
        ax = fig.add_subplot(gs[i, 2])
        m = spot_metrics(*probe.hits[i])
        label = "specimen plane" if off == 0 else f"{off:+.0f} um ({'under' if off < 0 else 'over'} focus)"
        show_psf(ax, img, window_um, f"{label}\nd50 = {m['d50_um']:.1f} um, rms = {m['rms_um']:.1f} um")
        axes.append(ax)
    return axes, dict(probe=probe, psfs=psfs, centre=centre, window_um=window_um)


def column_figure(column, s, planes_um=(-50.0, 0.0, 50.0), window_um=None, npix=96, title=None):
    """The main overview figure for one machine setting."""
    fig = plt.figure(figsize=(17, 9.5))
    gs = GridSpec(1, 1, figure=fig, left=0.01, right=0.97, top=0.92, bottom=0.04)
    _, info = draw_overview(fig, gs[0], column, s, planes_um, window_um, npix)
    fig.suptitle(title or settings_label(s), fontsize=11)
    return fig, info


# ------------------------------------------------------------------ grid figures
def sweep_figure(column, sweeps, base=None, window_um=40.0, npix=80):
    """One row per knob, one column per value: the PSF in the specimen plane.
    sweeps: list of (knob name, values, row label)."""
    from .column import SEMSettings
    base = base or SEMSettings()
    n_cols = max(len(v) for _, v, _ in sweeps)
    fig, axes = plt.subplots(len(sweeps), n_cols, figsize=(1.75 * n_cols + 1.2, 1.85 * len(sweeps)),
                             squeeze=False)
    for r, (key, values, label) in enumerate(sweeps):
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(values):
                ax.axis("off")
                continue
            s = base.replace(**{key: values[c]})
            pr = column.probe(s, (0.0,))
            psf, _ = probe_psfs(pr, window_um, npix)
            m = spot_metrics(*pr.hits[0])
            show_psf(ax, psf[0], window_um, f"{values[c]:g}   d50 {m['d50_um']:.1f} um")
            if c:
                ax.set_yticklabels([])
        axes[r, 0].set_ylabel(label, fontsize=8)
    fig.suptitle(f"PSF at the specimen plane while turning one knob at a time "
                 f"(window {window_um:g} um, others nominal)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def through_focus_figure(column, configs, planes_um=(-60, -30, 0, 30, 60), window_um=48.0, npix=80):
    """Rows: named machine settings. Columns: PSF at planes through focus + a spot diagram."""
    n_cols = len(planes_um) + 1
    fig, axes = plt.subplots(len(configs), n_cols, figsize=(1.9 * n_cols + 1.5, 1.95 * len(configs)),
                             squeeze=False)
    for r, (name, s) in enumerate(configs):
        pr = column.probe(s, planes_um)
        psfs, centre = probe_psfs(pr, window_um, npix)
        for c, (img, off) in enumerate(zip(psfs, pr.planes_um)):
            m = spot_metrics(*pr.hits[c])
            show_psf(axes[r, c], img, window_um,
                     (f"z {off:+.0f} um\n" if r == 0 else "") + f"d50 {m['d50_um']:.1f}")
        x, y = pr.hits[pr.plane_index(0)]
        k = np.linspace(0, len(x) - 1, min(3000, len(x))).astype(int)
        ax = axes[r, -1]
        ax.scatter((x[k] - centre[0]) * 1e3, (y[k] - centre[1]) * 1e3, s=0.3, c="k", alpha=0.4)
        ax.set_xlim(-window_um / 2, window_um / 2)
        ax.set_ylim(-window_um / 2, window_um / 2)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        if r == 0:
            ax.set_title("spot diagram\n(specimen plane)", fontsize=8)
        axes[r, 0].set_ylabel(name, fontsize=8)
    fig.suptitle("Through-focus PSFs (negative = under focus, positive = over focus)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def field_map_figure(column, configs, field_mm=1.0, n=5, window_um=40.0, npix=64):
    """PSF at an n x n grid of scan positions over +-field_mm, one panel per configuration.
    Each PSF is centred on its own centroid; the label gives the landing error (distortion)."""
    fig = plt.figure(figsize=(6.2 * len(configs), 7.4))
    subs = fig.subfigures(1, len(configs), squeeze=False)[0]
    pos = np.linspace(-field_mm, field_mm, n)
    for sub, (name, s) in zip(subs, configs):
        axes = sub.subplots(n, n)
        for iy, Y in enumerate(pos[::-1]):
            for ix, X in enumerate(pos):
                pr = column.probe(s.replace(scan_x_mm=X, scan_y_mm=Y), (0.0,))
                psf, c = probe_psfs(pr, window_um, npix)
                err = np.hypot(*(c - pr.target)) * 1e3
                show_psf(axes[iy, ix], psf[0], window_um)
                axes[iy, ix].set_title(f"({X:+.1f},{Y:+.1f}) mm, shift {err:.1f} um", fontsize=6)
                axes[iy, ix].set_xticks([])
                axes[iy, ix].set_yticks([])
        sub.suptitle(f"{name}: PSF across the scan field", fontsize=10)
    return fig


def sample_images_figure(column, configs, samples, fov_mm, n_field=3, kernel_px=41, dose=None):
    """Top row: the specimens. Following rows: their simulated SEM images per configuration."""
    from .imaging import scan_image, scan_kernels
    names = list(samples)
    n_px = samples[names[0]].shape[0]
    half = fov_mm * 1e3 / 2
    fig, axes = plt.subplots(len(configs) + 1, len(names), figsize=(3.1 * len(names), 3.0 * (len(configs) + 1)),
                             squeeze=False)
    for c, k in enumerate(names):
        show_psf(axes[0, c], samples[k], 2 * half, f"specimen: {k}", gamma=1)
    axes[0, 0].set_ylabel("ground truth", fontsize=9)
    for r, (name, s) in enumerate(configs, start=1):
        kernels = scan_kernels(column, s, fov_mm, n_px, n_field, kernel_px)
        for c, k in enumerate(names):
            img = scan_image(column, s, samples[k], fov_mm, kernels=kernels, dose=dose, rng=r)
            show_psf(axes[r, c], img, 2 * half, gamma=1)
        axes[r, 0].set_ylabel(name, fontsize=9)
    for ax in axes.flat:
        ax.tick_params(labelsize=6)
    fig.suptitle(f"Simulated SEM images of point-like specimens (field of view {fov_mm * 1e3:.0f} um, "
                 f"{n_px} px, field-dependent PSF on a {n_field}x{n_field} grid)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def database_examples_figure(db, n_examples=8, seed=0, crop_px=64):
    """Random database entries: through-focus PSFs, a crop of each sample image, and the knobs."""
    meta = db["meta"]
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(len(db["params"]), size=min(n_examples, len(db["params"])), replace=False))
    planes, snames = meta["planes_um"], meta["sample_names"]
    n_cols = len(planes) + len(snames) + 1
    fig, axes = plt.subplots(len(ids), n_cols, figsize=(1.9 * n_cols + 1.5, 2.0 * len(ids)), squeeze=False,
                             gridspec_kw=dict(width_ratios=[1] * (n_cols - 1) + [2.2]))
    n_px = meta["image_px"]
    lo = (n_px - crop_px) // 2
    crop_um = crop_px * meta["fov_mm"] * 1e3 / n_px
    for r, i in enumerate(ids):
        for c, off in enumerate(planes):
            show_psf(axes[r, c], db["psfs"][i, c], meta["psf_window_um"], f"PSF z {off:+.0f} um" if r == 0 else None)
        for j, k in enumerate(snames):
            img = db["images"][i, j, lo:lo + crop_px, lo:lo + crop_px].astype(float)
            show_psf(axes[r, len(planes) + j], img, crop_um, f"image: {k}\n(centre crop)" if r == 0 else None, gamma=1)
        axes[r, 0].set_ylabel(f"entry {i}", fontsize=8)
        txt = "\n".join(f"{k} = {v:.3g}" for k, v in zip(meta["param_names"], db["params"][i]))
        ax = axes[r, -1]
        ax.axis("off")
        ax.text(0, 0.5, txt, fontsize=5.5, va="center", family="monospace")
    for ax in axes[:, :-1].flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Database entries: machine settings -> through-focus PSF -> images", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def database_sensitivity_figure(db, metric="d50_um_+0um"):
    """Scatter of a spot metric against every knob: which knobs matter most for the probe."""
    meta = db["meta"]
    names = meta["param_names"]
    m = db["metrics"][:, meta["metric_names"].index(metric)]
    n_cols = 5
    n_rows = int(np.ceil(len(names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows), squeeze=False)
    for ax, (k, name) in zip(axes.flat, enumerate(names)):
        x = db["params"][:, k]
        ax.scatter(x, m, s=5, alpha=0.6)
        rho = np.corrcoef(np.abs(x - np.mean(meta["param_ranges"][name])), m)[0, 1]
        ax.set_title(f"{name}\n|corr| with distance from range centre: {abs(rho):.2f}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_ylabel(metric, fontsize=7)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.suptitle(f"Database: {metric} against each knob (all other knobs random)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def settings_label(s, defaults=None):
    """Short text listing the knobs that differ from their defaults."""
    from .column import SEMSettings
    d = (defaults or SEMSettings()).as_dict()
    diff = [f"{k}={v:.4g}" for k, v in s.as_dict().items() if not np.isclose(v, d[k])]
    return "nominal settings" if not diff else ", ".join(diff)
