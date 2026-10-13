# Static MEO GNSS Monitor

A standalone, static GNSS constellation-health dashboard for Galileo, BeiDou,
GPS, and GLONASS. It preserves the original monitor's health calculations,
plane/slot analysis, satellite plots, GPS legacy overlay, and single-/combined-
constellation DOP calculations, while loading only bundled data.

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
