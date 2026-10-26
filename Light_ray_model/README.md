# SEM digital twin: light-ray version

A 3D ray-traced model of an SEM column that uses **light rays and glass lenses** as a stand-in for
electrons and magnetic lenses. It is the first step towards a twin that maps
*machine settings → probe (PSF) → image*, so that an observed blur can be traced back to the
setting that caused it (instead of only fitting aberration coefficients, as in
`SEM_Aberrations_Report 1.pdf`).

Aberrations are **not put in by hand**: they come out of the geometry. Rays are traced with Snell's
law through real spherical glass surfaces, so spherical aberration, coma, astigmatism, field
aberrations and chromatic aberration appear when you turn the corresponding physical knob.

## Quick start

```
pip install -r requirements.txt
python scripts/view_3d.py            # interactive 3D viewer with sliders
python scripts/make_figures.py       # all overview figures -> output/figures
python scripts/build_database.py     # settings -> PSF -> image database -> output/database
python scripts/make_figures.py --only database
```

## The column

Layout after Fig. 1 of the report, made 3D (beam travels in +z, all lengths in mm):

|   z [mm]   | element                       | stand-in for                                          |
|------------|-------------------------------|-------------------------------------------------------|
|      0     | point/disc light source       | electron gun (source size ↔ d50,geo)                  |
|      30    | Condenser 1, biconvex f≈15 mm | condenser lens 1, forms a crossover                   |
|     100    | Condenser 2, biconvex f≈40 mm | condenser lens 2, roughly collimates                  |
|     120    | circular aperture             | final aperture (sets convergence angle θ)             |
|  140, 160  | two deflectors                | double-deflection scan coils (pivot in the objective) |
|     172    | thin quadrupole               | stigmator                                             |
|     190    | Objective, biconvex f≈15 mm   | objective lens                                        |
|   ≈206.4   | specimen plane                | specimen (placed at the nominal minimum-RMS focus)    |

## The knobs (`SEMSettings`)

| knob                                    | what it does                                    | aberration it produces                                  |
|-----------------------------------------|-------------------------------------------------|---------------------------------------------------------|
| `defocus_um`                            | moves the specimen stage                        | defocus (C_df)                                          |
| `obj_strength`                          | objective curvature multiplier (≈ lens current) | defocus                                                 |
| `aperture_radius_mm`                    | aperture size                                   | spherical aberration grows ~r³                          |
| `aperture_shift_x/y_mm`                 | off-centre aperture                             | coma-like asymmetric probe                              |
| `stig_x`, `stig_y`                      | stigmator quadrupole [1/mm]                     | two-fold astigmatism (A1)                               |
| `obj_tilt_x/y_mrad`, `obj_shift_x/y_mm` | misaligned objective                            | coma (B2), probe shift                                  |
| `c1_strength`, `c2_strength`            | condenser currents                              | beam convergence → focus shift, spot size               |
| `bandwidth_nm`                          | spectral width of the source (glass dispersion) | chromatic aberration (≈ Cc with energy spread)          |
| `source_radius_um`                      | source size                                     | geometric blur (≈ d50,geo)                              |
| `scan_x/y_mm`                           | where the probe is sent                         | off-axis coma, astigmatism, field curvature, distortion |

## What comes out

* **3D overview** (`01_column_*.png`, `view_3d.py`): the column with a traced ray fan (rays blocked
  by the aperture shown dashed), a close-up of the focus with the rays and the **PSF slices drawn as
  planes at their real heights**, and the 2D PSF of each slice.
* **Knob sweeps** (`02_knob_sweeps.png`): PSF in the specimen plane while turning one knob at a time.
* **Through focus** (`03_through_focus.png`): under / in / over-focus PSFs and spot diagrams per
  configuration, comparable to Fig. 2 of the report.
* **Field map** (`04_field_map.png`): how the PSF changes across the scan field.
* **Sample images** (`05_sample_images.png`): simulated SEM images of point-like specimens (point grid,
  random points, small particles), using a PSF that varies with scan position.
* **Database** (`output/database/sem_ray_db.npz` + `index.csv`): N random configurations (Latin
  hypercube over `PARAM_RANGES` in `semtwin/database.py`), each with its settings, a through-focus PSF
  stack, spot metrics and the image of every sample. `06/07_database_*.png` give an overview.

```python
from semtwin.database import load_database
db = load_database("output/database/sem_ray_db.npz")
db["params"]   # (N, n_knobs)        machine settings, names in db["meta"]["param_names"]
db["psfs"]     # (N, n_planes, h, w) PSF at db["meta"]["planes_um"] from the specimen
db["images"]   # (N, n_samples, H, W) simulated images of db["samples"]
db["metrics"]  # (N, k)              names in db["meta"]["metric_names"]
```

## Using it from Python

```python
from semtwin.column import SEMColumn, SEMSettings
from semtwin.psf import probe_psfs, spot_metrics

col = SEMColumn()
s = SEMSettings(stig_x=3e-4, defocus_um=20)
probe = col.probe(s, planes_um=(-50, 0, 50))   # trace ~131k rays
psfs, centre = probe_psfs(probe, window_um=40, npix=96)
print(spot_metrics(*probe.hits[1]))
```

## Code layout

```
semtwin/optics.py    rays, glass dispersion, lens / aperture / deflector / stigmator elements
semtwin/column.py    SEMSettings (knobs), ColumnGeometry (hardware), SEMColumn (trace, probe, calibration)
semtwin/psf.py       ray hits -> PSF image, spot metrics
semtwin/samples.py   point-like specimens
semtwin/imaging.py   scanning image formation with field-dependent PSF
semtwin/database.py  build / load the settings -> PSF -> image database
semtwin/plotting.py  3D column, 3D focus zoom, all figures
scripts/             view_3d.py, make_figures.py, build_database.py
```

## Limits of this version

* Geometric optics only: no diffraction, so an aberration-free probe would be a point (the source
  size and spherical aberration set the minimum spot here). The wave-optical step is the next one:
  the same column can supply the pupil phase for a Fourier-optics PSF (as in `WavePSF.py`).
* Glass lenses, not magnetic fields: aberration magnitudes are in the right spirit, not SEM values.
  Lengths are mm and spot sizes µm, rather than SEM nm.
* The image model is incoherent and linear (signal = specimen ⋆ probe); no electron-specimen
  interaction volume or detector physics.
* Images remove the uniform image shift a misaligned lens causes (as an operator would re-centre);
  shifts that change across the field (distortion) are kept.
