# Static MEO GNSS Monitor

A standalone, static GNSS constellation-health dashboard for Galileo, BeiDou,
GPS, and GLONASS. It provides plane/slot analysis, satellite plots, GPS legacy
overlay, and single-/combined-constellation DOP calculations, while loading only
bundled data.

This repository contains **only the static app**. There is no live app, orbital
data downloader, credential form, or provider refresh action. No Space-Track
or CelesTrak credentials are required, and none are stored here.

## Deploy on Streamlit Community Cloud

In Streamlit Community Cloud, create an app with these settings:

| Setting | Value |
| --- | --- |
| Repository | `ElectricPlant19/Static_MEO_GNSS_Monitor` |
| Branch | `main` |
| Main file path | `static_app.py` |
| Python version (Advanced settings) | `3.13` |
| Secrets | Leave empty; none are needed |

The repository-root `requirements.txt` pins the tested runtime dependencies.
The root `.streamlit/config.toml` disables usage telemetry. No system packages,
external data services, or local files from the original project are needed.
See the official [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

## Fixed snapshot and epoch policy

The bundled `static_snapshot.json` contains 125 genuine Space-Track historical
TLEs, including all 111 satellites marked `USABLE` in the frozen roster and
seven legacy GPS satellites. SGP4 propagates every orbit to the common analysis
epoch **2026-07-14 12:00:00 UTC** before geometry and DOP are calculated. Age
scores also use that epoch, so calendar time does not change the assessment.

| Constellation | Frozen roster | USABLE | Stored TLEs |
| --- | ---: | ---: | ---: |
| Galileo | 33 | 28 | 32 |
| BeiDou | 31 | 28 | 30 |
| GPS | 32 | 32 | 32 |
| GLONASS | 24 | 23 | 24 |

Raw TLE epochs are nearby observations, not identical timestamps: usable
elements are at most 31.126 hours from the common analysis epoch, and the full
set spans approximately 60.6 hours. The file is validated for integrity,
catalogue identity, raw TLE checksums, complete usable coverage, launch dates,
and a maximum 72-hour offset before the app runs. Invalid data stops the app;
there is no live-data fallback.

Two non-operational orbital entries are explicitly unavailable: Galileo
`GSAT0205` (no suitable history returned) and BeiDou `BD3-M3S` (a roster
placeholder without a NORAD ID). No synthetic orbit is substituted.

Service-status flags are frozen repository metadata captured with the snapshot,
not independently verified historical service statuses for July 2026. This
dashboard reproduces the project's health model at a fixed orbital epoch; it
is not an authoritative historical GNSS service-health report.

All four datasets are loaded at startup, so combined DOP requires no fetching.
The DOP map uses bundled land outlines rather than online tiles or CDNs. Click
its 2-degree observer grid, or enter exact latitude and longitude in the input
controls.

## Along-track and cross-track diagnostics

The **Along-Track** tab shows the estimated orbital-phase drift rate:

- **Estimated along-track rate (km/day)** is `2π R₀ (n_TLE − n₀)`, using the
  configured nominal radius in km and mean motions in rev/day. Positive means
  gaining orbital phase. This is a circular-orbit approximation; TLE mean motion
  stays at its own element epoch and is not an observed drift history.

The **Cross-Track** tab uses common-epoch SGP4 position and velocity in GCRS:

- The peer-plane normal is the normalized mean of **unit** angular-momentum
  vectors of the usable MEO satellites selected for RAAN clustering.
- **Peer cross-track (km)** is the signed projection `r · n̂_peer`. Its sign
  follows the prograde angular-momentum normal; averages use absolute distances.
- **Peer plane angle (degrees)** is the angle between `r × v` and the peer
  normal, calculated with `atan2(|ĥ × n̂_peer|, ĥ · n̂_peer)`. This stays nonzero
  for a tilted orbit even at a plane crossing where cross-track distance is zero.
- **Model** columns repeat the geometry against the configured nominal plane.
  GPS uses the existing SPS RAAN model propagated to the analysis epoch, with
  FK5/J2000 axes rotated into GCRS and the configured nominal inclination.
  The other constellations' 0°/120°/240° RAANs are schematic app references
  interpreted in GCRS, not epoch-defined operator orbits. Model deviations are
  diagnostic, not service-compliance determinations. The peer fit absorbs a
  tilt shared by its entire plane.

The sidebar exposes independent **drift-rate (°/day)** and **peer plane-angle (°)**
limits. The defaults of 2 in each quantity are
dashboard screening choices, not official constellation tolerances. Flags and
chart lines use these configured limits, including the boundary. Existing
constellation-health/slot-occupancy scoring remains separate.

This snapshot has one TLE per satellite: future propagation would be
a prediction, while historical drift measurements require multiple observations.
Full radial/along-track/cross-track orbit-error assessment additionally requires
an independent reference trajectory at matched epochs and in a matched frame.

References: [Orekit local orbital frames](https://www.orekit.org/site-orekit-13.1/apidocs/org/orekit/frames/LOFType.html),
[CelesTrak mean-element documentation](https://celestrak.org/NORAD/documentation/),
and [GPS SPS Performance Standard](https://www.gps.gov/sites/default/files/2025-07/2020-SPS-performance-standard.pdf)
(Tables 3.2-1 through 3.2-3; the existing GPS nominal-plane model).

## Baseline occupants and reserves

For Galileo, BeiDou and GLONASS, a usable auxiliary/spare within the existing
15° matching band of an otherwise vacant baseline position contributes to
baseline coverage. It appears as a green regular marker labeled with that
position and participates in the baseline count and spacing/geometry score.
Its original roster slot remains in tooltips and the spacing table, alongside
the detected position. Geometry-based matching does not formally reassign its
operator slot. Matches are constrained to the displayed plane and are one-to-one;
an extra satellite does not fill a position already occupied by a usable regular.

In the bundled Galileo Plane 1 snapshot, GSAT0232/E16 (roster A17) fills A08,
so the display counts **8 baseline occupants and no additional reserves**.
GPS continues to use its separate baseline/expansion occupancy model. The
snapshot is not modified by this classification.

## Run locally

Use Python 3.13 (the verified deployment version), from this repository's root:

```bash
python -m pip install -r requirements.txt
python -m streamlit run static_app.py
```

The app's data and map paths are resolved relative to its files, not the working
directory. No parent repository is needed.

## Validate and test offline

```bash
python validate_snapshot.py
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

Tests block outbound connections and provider requests while exercising all
four dashboards and combined DOP from an unrelated working directory. They
also check propagation of all 125 stored TLEs, fixed-epoch age scoring, and
rejection of corrupted, incomplete, or epoch-inconsistent data.

## Map attribution

Land geometry is [Natural Earth public-domain data](https://www.naturalearthdata.com/about/terms-of-use/),
bundled from the [Natural Earth vector dataset](https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_110m_land.geojson).
