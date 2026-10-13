import streamlit as st
import pandas as pd
import numpy as np
from datetime import UTC, datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path

try:
    from skyfield.api import EarthSatellite, load as skyfield_load
    try:
        from skyfield.framelib import itrs as skyfield_itrs
    except ImportError:
        skyfield_itrs = None
    SKYFIELD_AVAILABLE = True
except ImportError:
    EarthSatellite = None
    skyfield_load = None
    skyfield_itrs = None
    SKYFIELD_AVAILABLE = False

_SKYFIELD_TIMESCALE = None

try:
    from .snapshot_data import load_snapshot as load_static_snapshot
except ImportError:
    # Streamlit's test runner can execute this script from an unrelated cwd
    # without adding its directory to the import path.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from snapshot_data import load_snapshot as load_static_snapshot

STATIC_ANALYSIS_EPOCH = None


def apply_static_snapshot(snapshot):
    """Replace mutable live rosters with the exact roster stored in the snapshot."""
    global GPS_LEGACY_SATS
    for constellation_name, cfg in CONSTELLATION_CONFIG.items():
        dataset = snapshot["constellations"].get(constellation_name)
        if not dataset or not isinstance(dataset.get("roster"), dict) or not isinstance(dataset.get("tles"), dict):
            raise RuntimeError(f"Static snapshot is missing {constellation_name} roster or elements.")
        # Clustering updates plane hints in memory; preserve the frozen input.
        cfg["sats"] = {svn: dict(entry) for svn, entry in dataset["roster"].items()}
    legacy = snapshot.get("gps_legacy", {})
    if not isinstance(legacy.get("roster", {}), dict) or not isinstance(legacy.get("tles", {}), dict):
        raise RuntimeError("Static snapshot is missing the GPS legacy overlay data.")
    GPS_LEGACY_SATS = legacy["roster"]


def load_static_session_data(snapshot, analysis_epoch):
    """Populate all analysis inputs without exposing any network-backed UI action."""
    for constellation_name, dataset in snapshot["constellations"].items():
        st.session_state[f"tle_data_{constellation_name}"] = dataset["tles"]
        st.session_state[f"fetch_time_{constellation_name}"] = analysis_epoch
        launch_dates = {}
        for svn, date_value in dataset.get("launch_dates", {}).items():
            if not date_value:
                continue
            try:
                launch_dates[svn] = datetime.strptime(date_value, "%Y-%m-%d").replace(tzinfo=UTC)
            except (TypeError, ValueError):
                continue
        st.session_state[f"launch_dates_{constellation_name}"] = launch_dates
    st.session_state["legacy_tle_data_GPS"] = snapshot["gps_legacy"]["tles"]

# Rosters come exclusively from the bundled snapshot at startup.
GALILEO_SATS = {}
BEIDOU_SATS = {}
GPS_SATS = {}
GLONASS_SATS = {}
GPS_LEGACY_SATS = {}

# ============================
# ORBITAL CONSTANTS
# ============================

EARTH_RADIUS = 6371  # km

# --- Galileo nominal parameters ---
GAL_NOMINAL_MEAN_MOTION = 1.70475137  # rev/day
GAL_NOMINAL_INCLINATION = 56.0  # degrees
GAL_NOMINAL_ALTITUDE = 23229  # km
GAL_ORBITAL_RADIUS = EARTH_RADIUS + GAL_NOMINAL_ALTITUDE  # ~29600 km
GAL_KM_PER_DEGREE = (2 * np.pi * GAL_ORBITAL_RADIUS) / 360
GAL_DESIGN_LIFE = 12  # years

GAL_NOMINAL_RAAN = {
    1: 0.0,    # Plane A
    2: 120.0,  # Plane B
    3: 240.0   # Plane C
}

GAL_NOMINAL_SLOT_ANGLES = {
    'A01': 0, 'A02': 45, 'A03': 90, 'A04': 135, 'A05': 180, 'A06': 225, 'A07': 270, 'A08': 315,
    'B01': 0, 'B02': 45, 'B03': 90, 'B04': 135, 'B05': 180, 'B06': 225, 'B07': 270, 'B08': 315,
    'C01': 0, 'C02': 45, 'C03': 90, 'C04': 135, 'C05': 180, 'C06': 225, 'C07': 270, 'C08': 315,
    # Auxiliary slots (approximate positions based on actual deployment)
    'A12': 180, 'A17': 90,
    'B14': 180, 'B15': 270,
    'C12': 180
}

# --- BeiDou-3 MEO nominal parameters ---
BDS_NOMINAL_MEAN_MOTION = 1.86274782  # rev/day (derived from ~21,528 km altitude)
BDS_NOMINAL_INCLINATION = 55.0  # degrees
BDS_NOMINAL_ALTITUDE = 21528  # km
BDS_ORBITAL_RADIUS = EARTH_RADIUS + BDS_NOMINAL_ALTITUDE  # ~27899 km
BDS_KM_PER_DEGREE = (2 * np.pi * BDS_ORBITAL_RADIUS) / 360
BDS_DESIGN_LIFE = 10  # years

BDS_NOMINAL_RAAN = {
    1: 0.0,    # Plane A
    2: 120.0,  # Plane B
    3: 240.0   # Plane C
}

BDS_NOMINAL_SLOT_ANGLES = {
    'A01': 0, 'A02': 45, 'A03': 90, 'A04': 135, 'A05': 180, 'A06': 225, 'A07': 270, 'A08': 315,
    'B01': 0, 'B02': 45, 'B03': 90, 'B04': 135, 'B05': 180, 'B06': 225, 'B07': 270, 'B08': 315,
    'C01': 0, 'C02': 45, 'C03': 90, 'C04': 135, 'C05': 180, 'C06': 225, 'C07': 270, 'C08': 315,
}

# --- GPS nominal parameters ---
GPS_NOMINAL_MEAN_MOTION = 2.0056  # rev/day (12 h sidereal repeat, expressed in solar days)
GPS_NOMINAL_INCLINATION = 55.0  # degrees
GPS_ORBITAL_RADIUS = 26560  # km semi-major axis
GPS_NOMINAL_ALTITUDE = GPS_ORBITAL_RADIUS - EARTH_RADIUS  # ~20189 km
GPS_KM_PER_DEGREE = (2 * np.pi * GPS_ORBITAL_RADIUS) / 360
GPS_DESIGN_LIFE = 12  # years, mixed operational constellation baseline

# --- GPS plane identity (2020 SPS PS Tables 3.2-1 / 3.2-3) ---
# Plane NUMBER is permanently bound to the spec's plane LETTER: 1=A ... 6=F.
# Every per-plane table below (spacings, slot order, expandable slot) is keyed
# on that binding, so a satellite must be assigned to the plane whose *physical*
# orbit it occupies -- never to a plane named by current-RAAN rank.
GPS_PLANE_LETTERS = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}

# Table 3.2-1: RAAN of each plane at the defined epoch.
GPS_SPS_REFERENCE_EPOCH = datetime(2016, 12, 31, 23, 59, 43, tzinfo=UTC)
GPS_NOMINAL_RAAN_AT_EPOCH = {
    1: 288.85,  # Plane A
    2: 348.85,  # Plane B
    3: 48.85,   # Plane C
    4: 108.85,  # Plane D
    5: 168.85,  # Plane E
    6: 228.85,  # Plane F
}
# Table 3.2-3: nodal regression rate. All six planes regress together, so the
# whole constellation rotates rigidly and the planes stay 60 deg apart forever.
# Over a decade this is a ~140 deg shift -- which is exactly why any plane
# labelling scheme based on sorting current RAAN goes stale (one full plane
# every 60/0.0402 = 1493 days = 4.09 years).
GPS_NODAL_REGRESSION_DEG_PER_DAY = -0.0402

# A cluster centre further than this from its propagated nominal RAAN means
# either a plane is genuinely drifting out of spec or the epoch handling is
# broken. Planes are 60 deg apart, so this leaves a large margin.
GPS_PLANE_MATCH_WARN_DEG = 15.0

# Retained for CONSTELLATION_CONFIG (static, at-epoch values). Runtime code must
# use gps_nominal_plane_raans(analysis_epoch) instead -- see below.
GPS_NOMINAL_RAAN = dict(GPS_NOMINAL_RAAN_AT_EPOCH)


def gps_nominal_plane_raans(analysis_epoch):
    """
    Nominal RAAN of each GPS plane at `analysis_epoch`, per SPS PS Table 3.2-1
    propagated by the Table 3.2-3 nodal regression rate.

    This is the only drift-proof way to name a GPS plane: the letters A-F are
    permanently attached to physical orbit planes, and their RAANs are a known
    function of time.
    """
    analysis_epoch = ensure_utc_datetime(analysis_epoch)
    days = (analysis_epoch - GPS_SPS_REFERENCE_EPOCH).total_seconds() / 86400.0
    shift = GPS_NODAL_REGRESSION_DEG_PER_DAY * days
    return {
        plane_num: (raan + shift) % 360.0
        for plane_num, raan in GPS_NOMINAL_RAAN_AT_EPOCH.items()
    }

# --- GPS per-plane nominal geometry (2020 reference, Epoch 31-Dec-2016) ---
# Opposite planes share identical spacings in reversed order (A↔F, B↔E, C↔D)
GPS_OPPOSITE_PLANES = {1: 6, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
GPS_PLANE_PAIR_KEY = {1: 'AF', 2: 'BE', 3: 'CD', 4: 'CD', 5: 'BE', 6: 'AF'}

# Nominal spacings per plane (consecutive, from 2020 reference)
GPS_NOMINAL_SPACINGS_BY_PLANE = {
    1: [103.55, 30.13, 119.98, 106.34],
    2: [92.38, 31.04, 105.60, 130.98],
    3: [129.68, 98.11, 32.13, 100.08],
    4: [32.13, 98.09, 129.71, 100.07],
    5: [105.55, 31.09, 92.38, 130.98],
    6: [106.34, 119.98, 30.14, 103.54],
}

# Relative ideal slot angles per plane (cumulative sum of spacings, starting at 0)
GPS_IDEAL_SLOT_ANGLES_BY_PLANE = {
    1: [0.00, 103.55, 133.68, 253.66],
    2: [0.00, 92.38, 123.42, 229.02],
    3: [0.00, 129.68, 227.79, 259.92],
    4: [0.00, 32.13, 130.22, 259.93],
    5: [0.00, 105.55, 136.64, 229.02],
    6: [0.00, 106.34, 226.32, 256.46],
}

# Nominal sorted spacings per plane pair (for symmetry comparison)
GPS_NOMINAL_SORTED_SPACINGS_BY_PAIR = {
    'AF': [30.14, 103.55, 106.34, 119.98],
    'BE': [31.07, 92.38, 105.58, 130.98],
    'CD': [32.13, 98.10, 100.08, 129.70],
}

# Structural bounds (stable across all epochs)
GPS_TIGHT_PAIR_RANGE = (20.0, 45.0)
GPS_MAX_GAP_WARNING = 135.0
GPS_MAX_GAP_CRITICAL = 160.0
GPS_RESERVE_OFFSET_THRESHOLD = 20.0
GPS_SPACING_TOLERANCE = 12.0

# --- GPS 2020 SPS PS expandable slot spec (rows 32-54 of 2020.html) ---
# Each plane has exactly one baseline slot authorized to dissolve into an aft (A) +
# fore (F) pair at the specific signed offsets below. Runtime classifier decides
# whether a satellite occupies the baseline or one of these expansion positions.
GPS_EXPANDABLE_BASELINE_SLOT = {
    1: 'A02',   # A2 -> A02A / A02F
    2: 'B01',   # B1 -> B01A / B01F
    3: 'C04',   # C4 -> C04A / C04F
    4: 'D02',   # D2 -> D02A / D02F
    5: 'E03',   # E3 -> E03A / E03F
    6: 'F02',   # F2 -> F02A / F02F
}

GPS_EXPANSION_OFFSETS_BY_PLANE = {
    1: {'aft': -15.24, 'fore': +11.20},
    2: {'aft': -14.60, 'fore': +13.96},
    3: {'aft': -17.24, 'fore':  +7.46},
    4: {'aft':  -7.47, 'fore': +17.23},
    5: {'aft': -13.96, 'fore': +14.60},
    6: {'aft': -11.20, 'fore': +15.23},
}

# GPS_IDEAL_SLOT_ANGLES_BY_PLANE stores cumulative angles in the SAME positional
# order as GPS_NOMINAL_SPACINGS_BY_PLANE (the 2020.html "Consec. Pair" order).
# This map names each positional index for its slot so we can go label <-> angle.
GPS_BASELINE_SLOT_ORDER_BY_PLANE = {
    1: ['A01', 'A03', 'A04', 'A02'],
    2: ['B01', 'B02', 'B04', 'B03'],
    3: ['C01', 'C04', 'C03', 'C02'],
    4: ['D01', 'D04', 'D02', 'D03'],
    5: ['E01', 'E02', 'E04', 'E03'],
    6: ['F01', 'F02', 'F03', 'F04'],
}

# Classification tolerances for the geometry-based occupancy classifier.
# Baseline: generous band because slots are 30-130 deg apart.
# Expansion: tighter band because offsets are +/- 7-17 deg from the expandable
# baseline, so a sat >10 deg from an expansion position is more likely a drifted
# baseline than an intentional expansion.
GPS_BASELINE_CLASSIFY_TOL = 25.0
GPS_EXPANSION_CLASSIFY_TOL = 10.0

# Two satellites this close to the same position are both credibly occupying it,
# and geometry cannot say which one the Control Segment considers the occupant.
# Launch date settles it: the incumbent holds the slot, the newer one is the
# replacement waiting to take over (SPS PS 3.2 gives auxiliaries no slot).
# Kept below the smallest fore/aft offset in Table 3.2-2 (C4F, +7.46 deg) so a
# satellite sitting at an expansion position can never be read as a contender
# for that slot's centre.
GPS_OCCUPANT_TIE_DEG = 6.0

# --- Legacy (retired / standby) GPS satellite residency ---
# A legacy satellite this close to a plane's propagated nominal RAAN is still
# station-keeping in it. Matches the tolerance gps/find_plane_residents.py uses
# for the same judgement, so the app and the sweep tool agree.
GPS_LEGACY_HOLDING_DEG = 6.0

# Beyond this it is no longer meaningfully in the plane. Planes are 60 deg
# apart, so a quarter-plane offset is a different orbit, not a wide slot.
# Deliberately NOT GPS_PLANE_MATCH_WARN_DEG: that constant warns about a whole
# cluster centre drifting from nominal, which is a different question.
GPS_LEGACY_RESIDENT_DEG = 15.0

GPS_LEGACY_TIER_HOLDING = 'holding'
GPS_LEGACY_TIER_DRIFTING = 'drifting'
GPS_LEGACY_TIER_DEPARTED = 'departed'

# Legacy fallbacks for polar chart markers when per-plane data unavailable
GPS_IDEAL_SLOT_ANGLES = [0, 30, 135, 255]
GPS_NOMINAL_SPACINGS = [30, 105, 120, 105]


def _derive_gps_expected_positions():
    """Build {plane_num: {slot_label: angle_deg}} covering 4 baseline + 2 expansion
    positions per GPS plane, in the reference frame where the plane's first
    baseline slot sits at 0 deg. Called once at module load."""
    result = {}
    for plane_num in range(1, 7):
        angles = GPS_IDEAL_SLOT_ANGLES_BY_PLANE.get(plane_num) or []
        labels = GPS_BASELINE_SLOT_ORDER_BY_PLANE.get(plane_num) or []
        if not angles or len(angles) != len(labels):
            continue
        positions = {label: angle % 360 for label, angle in zip(labels, angles)}
        expandable = GPS_EXPANDABLE_BASELINE_SLOT.get(plane_num)
        offsets = GPS_EXPANSION_OFFSETS_BY_PLANE.get(plane_num) or {}
        if expandable in positions and offsets:
            base_angle = positions[expandable]
            positions[expandable + 'A'] = (base_angle + offsets.get('aft', 0)) % 360
            positions[expandable + 'F'] = (base_angle + offsets.get('fore', 0)) % 360
        result[plane_num] = positions
    return result


GPS_EXPECTED_POSITIONS_BY_PLANE = _derive_gps_expected_positions()

# Semantic slot->angle map (fixes an earlier positional-index bug where
# A02/A03/A04 were mis-mapped). Includes expansion labels so reference-slot
# lookups on an expansion satellite return the expected expansion angle.
GPS_NOMINAL_SLOT_ANGLES = {
    label: angle
    for plane_positions in GPS_EXPECTED_POSITIONS_BY_PLANE.values()
    for label, angle in plane_positions.items()
}

# --- GLONASS nominal parameters ---
GLO_NOMINAL_MEAN_MOTION = 24 / (11 + 16 / 60)  # rev/day from 11 h 16 min period
GLO_NOMINAL_INCLINATION = 64.8  # degrees
GLO_NOMINAL_ALTITUDE = 19100  # km
GLO_ORBITAL_RADIUS = EARTH_RADIUS + GLO_NOMINAL_ALTITUDE
GLO_KM_PER_DEGREE = (2 * np.pi * GLO_ORBITAL_RADIUS) / 360
GLO_DESIGN_LIFE = 7  # years, GLONASS-M baseline

GLO_NOMINAL_RAAN = {
    1: 0.0,
    2: 120.0,
    3: 240.0
}

GLO_NOMINAL_SLOT_ANGLES = {
    'A01': 0, 'A02': 45, 'A03': 90, 'A04': 135, 'A05': 180, 'A06': 225, 'A07': 270, 'A08': 315,
    'B01': 0, 'B02': 45, 'B03': 90, 'B04': 135, 'B05': 180, 'B06': 225, 'B07': 270, 'B08': 315,
    'C01': 0, 'C02': 45, 'C03': 90, 'C04': 135, 'C05': 180, 'C06': 225, 'C07': 270, 'C08': 315,
}

# Legacy aliases for backward compatibility (Galileo defaults)
NOMINAL_MEAN_MOTION = GAL_NOMINAL_MEAN_MOTION
NOMINAL_INCLINATION = GAL_NOMINAL_INCLINATION
NOMINAL_ALTITUDE = GAL_NOMINAL_ALTITUDE
ORBITAL_RADIUS = GAL_ORBITAL_RADIUS
KM_PER_DEGREE = GAL_KM_PER_DEGREE
NOMINAL_RAAN = GAL_NOMINAL_RAAN
NOMINAL_SLOT_ANGLES = GAL_NOMINAL_SLOT_ANGLES

# ============================
# CONSTELLATION CONFIG
# ============================

CONSTELLATION_CONFIG = {
    'Galileo': {
        'sats': GALILEO_SATS,
        'nominal_mm': GAL_NOMINAL_MEAN_MOTION,
        'nominal_inc': GAL_NOMINAL_INCLINATION,
        'nominal_alt': GAL_NOMINAL_ALTITUDE,
        'orbital_radius': GAL_ORBITAL_RADIUS,
        'km_per_deg': GAL_KM_PER_DEGREE,
        'design_life': GAL_DESIGN_LIFE,
        'nominal_raan': GAL_NOMINAL_RAAN,
        'slot_angles': GAL_NOMINAL_SLOT_ANGLES,
        'name': 'Galileo',
        'ideal_regular': 8,
        'ideal_auxiliary': 2,
        'regular_slots': 8,
        'ideal_angles': list(range(0, 360, 45)),
        'nominal_spacings': [45] * 8,
        'angular_tick': 45,
        'icon': '🇪🇺',
        'gradient': 'rgba(0,100,200,0.3), rgba(0,50,150,0.2)',
        'gradient_light': 'rgba(0,100,200,0.15), rgba(0,150,100,0.1)',
        'accent_color': '#1976D2',
    },
    'BeiDou': {
        'sats': BEIDOU_SATS,
        'nominal_mm': BDS_NOMINAL_MEAN_MOTION,
        'nominal_inc': BDS_NOMINAL_INCLINATION,
        'nominal_alt': BDS_NOMINAL_ALTITUDE,
        'orbital_radius': BDS_ORBITAL_RADIUS,
        'km_per_deg': BDS_KM_PER_DEGREE,
        'design_life': BDS_DESIGN_LIFE,
        'nominal_raan': BDS_NOMINAL_RAAN,
        'slot_angles': BDS_NOMINAL_SLOT_ANGLES,
        'name': 'BeiDou',
        'ideal_regular': 8,
        'ideal_auxiliary': 1,
        'regular_slots': 8,
        'ideal_angles': list(range(0, 360, 45)),
        'nominal_spacings': [45] * 8,
        'angular_tick': 45,
        'icon': '🇨🇳',
        'gradient': 'rgba(200,50,50,0.3), rgba(150,30,30,0.2)',
        'gradient_light': 'rgba(200,50,50,0.15), rgba(150,80,30,0.1)',
        'accent_color': '#C62828',
    },
    'GPS': {
        'sats': GPS_SATS,
        'nominal_mm': GPS_NOMINAL_MEAN_MOTION,
        'nominal_inc': GPS_NOMINAL_INCLINATION,
        'nominal_alt': GPS_NOMINAL_ALTITUDE,
        'orbital_radius': GPS_ORBITAL_RADIUS,
        'km_per_deg': GPS_KM_PER_DEGREE,
        'design_life': GPS_DESIGN_LIFE,
        'nominal_raan': GPS_NOMINAL_RAAN,
        'slot_angles': GPS_NOMINAL_SLOT_ANGLES,
        'name': 'GPS',
        'ideal_regular': 4,
        # GPS "extras" are expansion positions, not reserves. Choice P: 4 baseline
        # required for full count credit; expansion is a neutral bonus (existing
        # aux_score branch gives free 5 pts when ideal_auxiliary == 0).
        'ideal_auxiliary': 0,
        'regular_slots': 4,
        'ideal_angles': GPS_IDEAL_SLOT_ANGLES,
        'ideal_angles_by_plane': GPS_IDEAL_SLOT_ANGLES_BY_PLANE,
        'nominal_spacings': GPS_NOMINAL_SPACINGS,
        'nominal_spacings_by_plane': GPS_NOMINAL_SPACINGS_BY_PLANE,
        'opposite_planes': GPS_OPPOSITE_PLANES,
        'expandable_baseline_slot': GPS_EXPANDABLE_BASELINE_SLOT,
        'expansion_offsets_by_plane': GPS_EXPANSION_OFFSETS_BY_PLANE,
        'expected_positions_by_plane': GPS_EXPECTED_POSITIONS_BY_PLANE,
        'spacing_strategy': 'gps_nominal',
        'angular_tick': 30,
        'icon': 'GPS',
        'gradient': 'rgba(40,120,90,0.3), rgba(20,80,140,0.2)',
        'gradient_light': 'rgba(40,120,90,0.15), rgba(20,80,140,0.1)',
        'accent_color': '#2E7D32',
    },
    'GLONASS': {
        'sats': GLONASS_SATS,
        'nominal_mm': GLO_NOMINAL_MEAN_MOTION,
        'nominal_inc': GLO_NOMINAL_INCLINATION,
        'nominal_alt': GLO_NOMINAL_ALTITUDE,
        'orbital_radius': GLO_ORBITAL_RADIUS,
        'km_per_deg': GLO_KM_PER_DEGREE,
        'design_life': GLO_DESIGN_LIFE,
        'nominal_raan': GLO_NOMINAL_RAAN,
        'slot_angles': GLO_NOMINAL_SLOT_ANGLES,
        'name': 'GLONASS',
        'ideal_regular': 8,
        'ideal_auxiliary': 0,
        'regular_slots': 8,
        'ideal_angles': list(range(0, 360, 45)),
        'nominal_spacings': [45] * 8,
        'angular_tick': 45,
        'icon': 'GLONASS',
        'gradient': 'rgba(120,70,170,0.3), rgba(60,110,160,0.2)',
        'gradient_light': 'rgba(120,70,170,0.15), rgba(60,110,160,0.1)',
        'accent_color': '#6A1B9A',
    }
}


def get_satellite_type(slot, plane, regular_slot_count=8):
    """Determine satellite type based on slot and plane"""
    if slot == 'Unk':
        return 'Spare'
    if plane == 0 or slot == 'Ext':
        return 'Eccentric'
    try:
        slot_num = int(slot[1:])
    except (ValueError, IndexError):
        return 'Spare' if plane > 0 else 'Eccentric'
    return 'Auxiliary' if slot_num > regular_slot_count else 'Regular'


def assign_clusters_to_known_planes(clusters, cluster_centers, active_sats, plane_numbers):
    """
    Map RAAN clusters back to constellation plane numbers using known slot metadata.
    Unknown-slot spares then inherit the plane of the cluster they physically occupy.
    """
    plane_numbers = list(plane_numbers)
    plane_number_set = set(plane_numbers)
    plane_votes = {}

    for cluster_id, sat_list in clusters.items():
        votes = {}
        for data in sat_list:
            sat_info = active_sats.get(data['svn'], {})
            slot = str(sat_info.get('slot', ''))
            known_plane = sat_info.get('plane', 0)
            if known_plane in plane_number_set and slot not in ('Unk', 'Ext'):
                votes[known_plane] = votes.get(known_plane, 0) + 1
        plane_votes[cluster_id] = votes

    cluster_to_plane = {}
    assigned_clusters = set()
    assigned_planes = set()

    vote_candidates = []
    for cluster_id, votes in plane_votes.items():
        for plane_num, vote_count in votes.items():
            vote_candidates.append((cluster_id, plane_num, vote_count, len(clusters.get(cluster_id, []))))

    vote_candidates.sort(key=lambda item: (-item[2], -item[3], item[1]))
    for cluster_id, plane_num, _, _ in vote_candidates:
        if cluster_id in assigned_clusters or plane_num in assigned_planes:
            continue
        cluster_to_plane[cluster_id] = plane_num
        assigned_clusters.add(cluster_id)
        assigned_planes.add(plane_num)

    remaining_clusters = [
        cluster_id
        for cluster_id in clusters.keys()
        if cluster_id not in assigned_clusters
    ]
    remaining_planes = [
        plane_num
        for plane_num in plane_numbers
        if plane_num not in assigned_planes
    ]

    for cluster_id, plane_num in zip(
        sorted(remaining_clusters, key=lambda cid: cluster_centers[cid]),
        remaining_planes
    ):
        cluster_to_plane[cluster_id] = plane_num

    return cluster_to_plane, plane_votes


def assign_clusters_by_nominal_raan(cluster_centers, plane_numbers, nominal_raan_by_plane):
    """
    Map RAAN clusters to plane numbers by matching each cluster centre to the
    plane whose *propagated nominal* RAAN it sits closest to.

    This replaces roster votes and RAAN-rank sorting for GPS. Both of those are
    unstable: the plane letters A-F are permanently attached to physical orbit
    planes, but their RAANs regress ~0.0402 deg/day, so any labelling derived
    from the current RAAN *ordering* rotates by a full plane every ~4.09 years.
    Matching against the propagated nominal is self-correcting at every epoch.

    Returns
    -------
    (cluster_to_plane, diagnostics)
        diagnostics: {'offsets': {plane_num: signed deviation of the cluster
        centre from nominal, deg}, 'max_offset': float}
    """
    # accepts either {cluster_id: centre} or a list indexed by cluster id
    centres = (cluster_centers if isinstance(cluster_centers, dict)
               else dict(enumerate(cluster_centers)))

    candidates = [
        (circular_distance_degrees(centre, nominal_raan_by_plane[plane_num]),
         cluster_id, plane_num)
        for cluster_id, centre in centres.items()
        for plane_num in plane_numbers
        if nominal_raan_by_plane.get(plane_num) is not None
    ]
    candidates.sort(key=lambda item: item[0])

    cluster_to_plane = {}
    taken_planes = set()
    offsets = {}
    for _distance, cluster_id, plane_num in candidates:
        if cluster_id in cluster_to_plane or plane_num in taken_planes:
            continue
        cluster_to_plane[cluster_id] = plane_num
        taken_planes.add(plane_num)
        offsets[plane_num] = circular_delta_degrees(
            centres[cluster_id], nominal_raan_by_plane[plane_num]
        )

    diagnostics = {
        'offsets': offsets,
        'max_offset': max((abs(v) for v in offsets.values()), default=0.0),
    }
    return cluster_to_plane, diagnostics


def fit_plane_phase_to_nominal(relative_angles, expected_angles,
                               step_deg=0.25, residual_cap_deg=30.0,
                               occupant_tol_deg=25.0):
    """
    Find the phase offset that best aligns a plane's satellites onto its nominal
    positions, replacing "anchor everything on one reference satellite".

    Anchoring on a reference satellite forces that satellite's deviation to be
    exactly zero and hides any common-mode in-plane error across the whole
    plane; the answer also changes depending on which satellite gets picked.
    Fitting a phase against the nominal geometry has neither problem.

    Cost is summed over satellites: each scores the squared distance to its
    nearest nominal position, saturating at `residual_cap_deg` so a genuine
    auxiliary far from every position cannot drag the fit.

    Two objectives were tried and rejected, both of which pick a phase ~115 deg
    wrong on the live plane B:
      * cost over nominal *positions* (each position to its nearest satellite) --
        an expanded slot's centre is legitimately vacant, so it contributes a
        large residual at the correct phase and penalises the truth;
      * greedy one-to-one satellite/position matching -- it forces a satellite
        onto a vacant centre at absurd distance, and rewards any phase that
        matches more satellites over one that matches fewer but far better.

    The residual defect of the satellite-based cost is that a replacement parked
    a few degrees from an incumbent double-counts that position and pulls the
    phase (measured: 1.00 deg on plane B, which shifted every deviation in the
    plane by 1.00 deg). That is handled by refitting: fit once with everything,
    use that phase to decide which satellites actually occupy a position, then
    refit on those alone. Auxiliaries then have no influence at all, so the frame
    no longer moves when one appears or disappears.

    Parameters
    ----------
    relative_angles : iterable of float
        In-plane angles relative to an arbitrary origin (deg).
    expected_angles : iterable of float
        The plane's nominal positions in the same frame the offset maps into.
    occupant_tol_deg : float
        How close a satellite must be, at the first-pass phase, to be treated as
        occupying a position and therefore included in the refit.

    Returns
    -------
    float : offset in degrees to add to `relative_angles`, or 0.0 if undefined.
    """
    relative_angles = [float(a) % 360.0 for a in relative_angles]
    expected_angles = [float(a) % 360.0 for a in expected_angles]
    if not relative_angles or not expected_angles:
        return 0.0

    cap_sq = residual_cap_deg ** 2
    steps = int(round(360.0 / step_deg))

    def best_phase(angles):
        best_offset, best_cost = 0.0, float('inf')
        for step in range(steps):
            offset = step * step_deg
            cost = 0.0
            for angle in angles:
                shifted = angle + offset
                nearest = min(
                    circular_distance_degrees(shifted, expected)
                    for expected in expected_angles
                )
                cost += min(nearest ** 2, cap_sq)
                if cost >= best_cost:
                    break
            if cost < best_cost:
                best_cost, best_offset = cost, offset
        return best_offset % 360.0

    coarse = best_phase(relative_angles)

    # Which satellites plausibly occupy a position at that phase? One satellite
    # per position, closest first, so a co-located replacement is left out.
    pairs = sorted(
        (circular_distance_degrees(angle + coarse, expected), sat_idx, pos_idx)
        for sat_idx, angle in enumerate(relative_angles)
        for pos_idx, expected in enumerate(expected_angles)
    )
    occupied_sats, taken_positions = {}, set()
    for deviation, sat_idx, pos_idx in pairs:
        if deviation > occupant_tol_deg:
            break
        if sat_idx in occupied_sats or pos_idx in taken_positions:
            continue
        occupied_sats[sat_idx] = pos_idx
        taken_positions.add(pos_idx)

    occupants = [relative_angles[i] for i in sorted(occupied_sats)]
    if len(occupants) < 2 or len(occupants) == len(relative_angles):
        return coarse
    return best_phase(occupants)


def get_nominal_angle(slot, slot_angles=None):
    """Get nominal in-plane angle for a slot."""
    if slot_angles is None:
        slot_angles = GAL_NOMINAL_SLOT_ANGLES
    return slot_angles.get(slot, None)


def circular_delta_degrees(angle, reference):
    """Return angle - reference normalized to [-180, 180]."""
    delta = (angle - reference + 180) % 360 - 180
    return delta


def circular_distance_degrees(angle, reference):
    """Return the absolute circular distance between two degree values."""
    return abs(circular_delta_degrees(angle, reference))


def plane_relative_argument_latitude(argument_latitude, ref_argument_latitude):
    """Argument of latitude expressed relative to a plane's reference satellite.

    Step one of the two-step transform that places a satellite in its plane's
    fitted frame. Shared by the active-satellite render path and the legacy
    overlay so the two can never drift apart -- a mismatch would silently put
    legacy markers in a different frame from the real satellites.
    """
    return circular_delta_degrees(argument_latitude, ref_argument_latitude)


def plane_plot_angle(relative_argument_latitude, plot_angle_offset):
    """Polar-chart angle for a relative argument of latitude.

    Step two of the transform described in plane_relative_argument_latitude.
    """
    return (relative_argument_latitude + plot_angle_offset) % 360


def classify_legacy_residency(raan_offset_deg):
    """Bucket a legacy satellite by how far its RAAN is from its nearest plane."""
    magnitude = abs(raan_offset_deg)
    if magnitude <= GPS_LEGACY_HOLDING_DEG:
        return GPS_LEGACY_TIER_HOLDING
    if magnitude <= GPS_LEGACY_RESIDENT_DEG:
        return GPS_LEGACY_TIER_DRIFTING
    return GPS_LEGACY_TIER_DEPARTED


def assign_legacy_planes(legacy_states, analysis_epoch):
    """Attribute each legacy satellite to the GPS plane it is nearest.

    Plane letters A-F are permanently attached to physical orbit planes whose
    RAANs regress at a known rate, so nominal RAANs must be propagated to the
    analysis epoch before comparison -- comparing against the 2016 table values
    would rotate every assignment by a full plane every 1493 days.

    Departed satellites get plane=None: a >15 deg offset means the object sits
    between planes, and every consumer must be able to keep it off a polar chart
    whose geometry assumes coplanarity.

    Returns {key: {'plane', 'nearest_plane', 'raan_offset_deg', 'tier'}}.
    """
    nominal_raans = gps_nominal_plane_raans(analysis_epoch)
    assignments = {}

    for key, state in legacy_states.items():
        raan = state.get('raan')
        if raan is None:
            continue

        best_plane = None
        best_offset = None
        for plane_num, nominal in nominal_raans.items():
            offset = circular_delta_degrees(raan, nominal)
            if best_offset is None or abs(offset) < abs(best_offset):
                best_plane = plane_num
                best_offset = offset

        if best_plane is None:
            continue

        tier = classify_legacy_residency(best_offset)
        assignments[key] = {
            'plane': None if tier == GPS_LEGACY_TIER_DEPARTED else best_plane,
            'nearest_plane': best_plane,
            'raan_offset_deg': best_offset,
            'tier': tier,
        }

    return assignments


def assign_expected_positions_by_geometry(sat_angles_sorted, ideal_angles):
    """
    Assign plotted satellites to the nominal markers shown in the polar chart.

    Some catalog slot labels identify constellation slots, but they do not always
    increase in plotted in-plane angle order. The spacing table should score the
    same nominal geometry that is drawn on the chart, so full planes are assigned
    one-to-one to the best ordered rotation of the nominal markers. Partial planes
    fall back to the nearest marker for each satellite.
    """
    if not sat_angles_sorted:
        return {}

    nominal_positions = sorted(float(angle) % 360 for angle in ideal_angles)
    if not nominal_positions:
        return {}

    if len(sat_angles_sorted) == len(nominal_positions):
        best_positions = None
        best_error = float('inf')

        for shift in range(len(nominal_positions)):
            candidate_positions = nominal_positions[shift:] + nominal_positions[:shift]
            error = sum(
                circular_delta_degrees(sat['Angle'], candidate_positions[idx]) ** 2
                for idx, sat in enumerate(sat_angles_sorted)
            )

            if error < best_error:
                best_error = error
                best_positions = candidate_positions

        return {
            sat['SVN']: best_positions[idx]
            for idx, sat in enumerate(sat_angles_sorted)
        }

    return {
        sat['SVN']: min(
            nominal_positions,
            key=lambda angle: circular_distance_degrees(sat['Angle'], angle)
        )
        for sat in sat_angles_sorted
    }


def assign_polar_marker_radii(plot_df, angle_col='Plot_Angle', output_col='Marker_Radius', collision_threshold_deg=2.0):
    """Stagger polar markers with nearly identical angles so reserves do not hide regular satellites."""
    if angle_col not in plot_df.columns:
        plot_df[output_col] = 1.0
        return plot_df

    result = plot_df.copy()
    result[output_col] = 1.0
    if result.empty:
        return result
    sorted_indices = sorted(
        result.index,
        key=lambda idx: float(result.at[idx, angle_col]) % 360
    )
    if not sorted_indices:
        return result

    groups = []
    current_group = [sorted_indices[0]]

    for idx in sorted_indices[1:]:
        previous_idx = current_group[-1]
        previous_angle = float(result.at[previous_idx, angle_col]) % 360
        current_angle = float(result.at[idx, angle_col]) % 360
        if circular_distance_degrees(current_angle, previous_angle) <= collision_threshold_deg:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
    groups.append(current_group)

    if (
        len(groups) > 1
        and circular_distance_degrees(
            float(result.at[groups[0][0], angle_col]) % 360,
            float(result.at[groups[-1][-1], angle_col]) % 360
        ) <= collision_threshold_deg
    ):
        groups[0] = groups[-1] + groups[0]
        groups.pop()

    type_priority = {'Regular': 0, 'Auxiliary': 1, 'Spare': 1}
    radius_offsets = [0.0, 0.08, -0.08, 0.15, -0.15, 0.21, -0.21]

    for group in groups:
        if len(group) <= 1:
            continue

        ordered_group = sorted(
            group,
            key=lambda idx: (
                type_priority.get(result.at[idx, 'Type'], 2),
                str(result.at[idx, 'Slot']),
                str(result.at[idx, 'SVN'])
            )
        )
        for order, idx in enumerate(ordered_group):
            offset = radius_offsets[order] if order < len(radius_offsets) else 0.21
            result.at[idx, output_col] = 1.0 + offset

    return result


def choose_plane_reference_svn(plane_df, active_sats, regular_slot_count=8, slot_angles=None,
                               excluded_slot=None):
    """Pick a stable slotted reference satellite, preferring regular baseline slots.

    excluded_slot: slot label whose position is ambiguous (GPS expandable slot);
    demoted below every other baseline candidate, used only as a last resort
    before auxiliaries.
    """
    if plane_df.empty or 'SVN' not in plane_df.columns:
        return None

    angle_col = next(
        (col for col in plane_df.columns if str(col).startswith('Argument of Latitude')),
        None
    )
    if angle_col is None:
        angle_col = next(
            (col for col in plane_df.columns if str(col).startswith('Mean Anomaly')),
            None
        )

    candidates = []
    for row_order, (_, row) in enumerate(plane_df.iterrows()):
        svn = row['SVN']
        sat_info = active_sats.get(svn, {})
        slot = sat_info.get('slot', '')
        plane = sat_info.get('plane', 0)
        sat_type = get_satellite_type(slot, plane, regular_slot_count)
        nominal_angle = get_nominal_angle(slot, slot_angles)

        if excluded_slot is not None and slot == excluded_slot:
            priority = 2
        elif sat_type == 'Regular' and nominal_angle is not None:
            priority = 0
        elif sat_type == 'Regular':
            priority = 1
        elif sat_type == 'Auxiliary' and nominal_angle is not None:
            priority = 3
        elif sat_type == 'Auxiliary':
            priority = 4
        else:
            priority = 5

        slot_number = 999
        try:
            slot_number = int(str(slot)[1:])
        except (TypeError, ValueError, IndexError):
            pass

        in_plane_angle = row.get(angle_col, 0) if angle_col else 0
        candidates.append((priority, slot_number, in_plane_angle, row_order, svn))

    return sorted(candidates)[0][-1] if candidates else None


def format_satellite_mix(regular_count, reserve_count):
    """Format regular/reserve counts without implying reserves exist when they do not."""
    if reserve_count > 0 and regular_count > 0:
        return f"{regular_count}+{reserve_count}", "Regular+Reserve"
    if reserve_count > 0:
        return f"{reserve_count}", "Reserve only"
    return f"{regular_count}", "Regular only"


def format_target_mix(regular_count, reserve_count):
    """Format the nominal plane target without requiring optional reserves."""
    if reserve_count > 0:
        return f"{regular_count}+{reserve_count}"
    return f"{regular_count}"


def calculate_coverage_spacing_health(regular_count, ideal_regular):
    """Score constellations whose plane health is defined by occupied baseline slots."""
    if ideal_regular <= 0:
        return 100.0
    return min(100.0, (regular_count / ideal_regular) * 100.0)


def analyze_gps_spacing_pattern(actual_spacings, plane_num,
                                 tolerance=GPS_SPACING_TOLERANCE,
                                 nominal_spacings_by_plane=None,
                                 explicit_nominal=None):
    """
    Validate actual satellite spacings against per-plane nominal pattern.

    Parameters
    ----------
    actual_spacings : list of float
        Consecutive spacing values (ascending angle order, circular).
    plane_num : int
        Plane number (1-6 for GPS).
    tolerance : float
        Acceptable deviation per spacing in degrees (default GPS_SPACING_TOLERANCE).
    nominal_spacings_by_plane : dict or None
        Per-plane nominal spacings. Falls back to GPS_NOMINAL_SPACINGS_BY_PLANE.
    explicit_nominal : list of float or None
        Runtime-generated N-value nominal (e.g. from occupancy for expanded planes).
        Overrides the per-plane lookup when provided.

    Returns
    -------
    dict with keys:
        best_rotation, matched_nominal, per_spacing_deviation,
        rmse, pattern_score, tight_pair_found, tight_pair_value,
        max_gap, max_gap_status
    """
    if explicit_nominal is not None:
        nominal = list(explicit_nominal)
    else:
        if nominal_spacings_by_plane is None:
            nominal_spacings_by_plane = GPS_NOMINAL_SPACINGS_BY_PLANE
        nominal = nominal_spacings_by_plane.get(plane_num, GPS_NOMINAL_SPACINGS)

    if not actual_spacings or len(actual_spacings) != len(nominal):
        return {
            'best_rotation': 0,
            'matched_nominal': nominal,
            'per_spacing_deviation': [0.0] * len(nominal),
            'rmse': 0.0,
            'pattern_score': 0.0,
            'tight_pair_found': False,
            'tight_pair_value': None,
            'max_gap': 0.0,
            'max_gap_status': 'ok',
        }

    best_rotation = 0
    best_rmse = float('inf')
    best_matched = list(nominal)

    for shift in range(len(nominal)):
        rotated = nominal[shift:] + nominal[:shift]
        deviations = [
            (actual_spacings[i] - rotated[i] + 180) % 360 - 180
            for i in range(len(actual_spacings))
        ]
        rmse = np.sqrt(np.mean(np.array(deviations) ** 2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_rotation = shift
            best_matched = rotated

    per_deviation = [
        (actual_spacings[i] - best_matched[i] + 180) % 360 - 180
        for i in range(len(actual_spacings))
    ]
    pattern_score = 100.0 * np.exp(-0.004 * (best_rmse ** 2))

    tight_pair_values = [s for s in actual_spacings if GPS_TIGHT_PAIR_RANGE[0] <= s <= GPS_TIGHT_PAIR_RANGE[1]]
    tight_pair_found = len(tight_pair_values) == 1
    tight_pair_value = tight_pair_values[0] if tight_pair_values else None

    max_gap = max(actual_spacings)
    if max_gap >= GPS_MAX_GAP_CRITICAL:
        max_gap_status = 'critical'
    elif max_gap >= GPS_MAX_GAP_WARNING:
        max_gap_status = 'warning'
    else:
        max_gap_status = 'ok'

    return {
        'best_rotation': best_rotation,
        'matched_nominal': best_matched,
        'per_spacing_deviation': per_deviation,
        'rmse': best_rmse,
        'pattern_score': pattern_score,
        'tight_pair_found': tight_pair_found,
        'tight_pair_value': tight_pair_value,
        'max_gap': max_gap,
        'max_gap_status': max_gap_status,
    }


def analyze_gps_plane_symmetry(plane_spacings_dict, opposite_planes=None):
    """
    Compare spacing vectors between opposite GPS plane pairs.

    Parameters
    ----------
    plane_spacings_dict : dict
        {plane_num: [sorted_spacings]} for all analyzed planes.
    opposite_planes : dict or None
        {plane_num: opposite_plane_num} mapping. Falls back to GPS_OPPOSITE_PLANES.

    Returns
    -------
    dict with keys:
        pair_scores: {pair_key: {deviation, score, spacings_a, spacings_b, details}}
        overall_score: float (0-100)
    """
    if opposite_planes is None:
        opposite_planes = GPS_OPPOSITE_PLANES

    pair_scores = {}
    processed_pairs = set()

    for plane_num in sorted(plane_spacings_dict.keys()):
        opp_plane = opposite_planes.get(plane_num)
        if opp_plane is None or opp_plane in processed_pairs:
            continue

        pair_key = GPS_PLANE_PAIR_KEY.get(plane_num, f'{plane_num}v{opp_plane}')
        processed_pairs.add(plane_num)
        processed_pairs.add(opp_plane)

        spacings_a = sorted(plane_spacings_dict.get(plane_num, []))
        spacings_b = sorted(plane_spacings_dict.get(opp_plane, []))

        if not spacings_a or not spacings_b:
            continue

        min_len = min(len(spacings_a), len(spacings_b))
        deviations = [
            abs(spacings_a[i] - spacings_b[i])
            for i in range(min_len)
        ]
        mean_dev = np.mean(deviations) if deviations else 0.0
        score = 100.0 * np.exp(-0.02 * (mean_dev ** 2))

        pair_scores[pair_key] = {
            'plane_nums': (plane_num, opp_plane),
            'deviation': mean_dev,
            'score': score,
            'spacings_a': spacings_a[:min_len],
            'spacings_b': spacings_b[:min_len],
            'details': deviations,
        }

    overall_score = np.mean([v['score'] for v in pair_scores.values()]) if pair_scores else 0.0

    return {
        'pair_scores': pair_scores,
        'overall_score': overall_score,
    }


def analyze_gps_reserve_proximity(plot_df, regular_sats_sorted,
                                   offset_threshold=GPS_RESERVE_OFFSET_THRESHOLD):
    """
    Identify reserve satellites within offset threshold of baseline slots.

    Parameters
    ----------
    plot_df : pd.DataFrame
        DataFrame with all satellites (must have 'SVN', 'Slot', 'Type', 'Plot_Angle').
    regular_sats_sorted : list of dict
        Regular satellites sorted by angle, each with 'SVN', 'Slot', 'Angle'.
    offset_threshold : float
        Max angular offset to consider co-located.

    Returns
    -------
    list of dict with keys:
        reserve_svn, reserve_slot, reserve_angle, nearest_baseline_slot,
        nearest_baseline_svn, nearest_baseline_angle, offset_deg, is_collocated
    """
    reserves = []
    reserve_df = plot_df[plot_df['Type'] != 'Regular'].copy()

    if reserve_df.empty or not regular_sats_sorted:
        return reserves

    for _, res_row in reserve_df.iterrows():
        res_angle = res_row.get('Plot_Angle', 0) % 360
        nearest = min(
            regular_sats_sorted,
            key=lambda rs: min(
                abs(res_angle - rs['Angle']),
                360 - abs(res_angle - rs['Angle'])
            )
        )
        offset = (res_angle - nearest['Angle'] + 180) % 360 - 180
        is_collocated = abs(offset) <= offset_threshold

        reserves.append({
            'reserve_svn': res_row['SVN'],
            'reserve_slot': res_row.get('Slot', 'Unk'),
            'reserve_angle': res_angle,
            'nearest_baseline_slot': nearest['Slot'],
            'nearest_baseline_svn': nearest['SVN'],
            'nearest_baseline_angle': nearest['Angle'],
            'offset_deg': round(offset, 2),
            'is_collocated': is_collocated,
        })

    return reserves


def _resolve_gps_occupant_ties(sats, pos_items, assigned_sat, assigned_pos,
                               launch_dates, expansion_labels,
                               baseline_tol, expansion_tol,
                               tie_deg=GPS_OCCUPANT_TIE_DEG):
    """
    Hand a contested position to its longest-serving satellite, in place.

    A challenge is only entertained when *both* satellites are within `tie_deg`
    of the position, so an older satellite that has genuinely drifted cannot win
    a slot on age alone. Satellites with no known launch date never challenge and
    are never displaced, so the result degrades to pure geometry.

    A displaced satellite becomes unassigned and may itself challenge a later
    position, which is how a chain of replacements resolves.
    """
    def launched(sat_idx):
        return launch_dates.get(sats[sat_idx]['svn'])

    for pos_idx, holder_idx in list(assigned_pos.items()):
        label, pos_angle = pos_items[pos_idx]
        tol = expansion_tol if label in expansion_labels else baseline_tol
        gate = min(tol, tie_deg)

        if abs(circular_delta_degrees(sats[holder_idx]['angle'], pos_angle)) > gate:
            continue
        holder_launch = launched(holder_idx)
        if holder_launch is None:
            continue

        challenger_idx, challenger_launch = None, holder_launch
        for sat_idx in range(len(sats)):
            if sat_idx in assigned_sat:
                continue
            if abs(circular_delta_degrees(sats[sat_idx]['angle'], pos_angle)) > gate:
                continue
            when = launched(sat_idx)
            if when is not None and when < challenger_launch:
                challenger_idx, challenger_launch = sat_idx, when

        if challenger_idx is None:
            continue

        signed = circular_delta_degrees(sats[challenger_idx]['angle'], pos_angle)
        del assigned_sat[holder_idx]
        assigned_sat[challenger_idx] = (pos_idx, signed, abs(signed))
        assigned_pos[pos_idx] = challenger_idx


def build_gps_plane_occupancy(plot_df, plane_num,
                              baseline_tol=GPS_BASELINE_CLASSIFY_TOL,
                              expansion_tol=GPS_EXPANSION_CLASSIFY_TOL,
                              launch_dates=None):
    """
    Classify every USABLE satellite in a GPS plane against the plane's 6 expected
    positions (4 baseline + 2 expansion) using greedy minimum-distance bipartite
    assignment. Single source of truth for GPS per-plane geometry: consumed by the
    spacing analysis, polar chart, plane health count, and expansion UI expander.

    Parameters
    ----------
    plot_df : pd.DataFrame
        Per-plane frame that already carries 'SVN', 'PRN', 'Slot', 'Status',
        'Plot_Angle' columns. Only rows with Status == 'USABLE' participate.
    plane_num : int
        GPS plane number 1-6.
    baseline_tol, expansion_tol : float
        Max |deviation deg| for a sat-to-position pair to be accepted as a
        baseline/expansion match respectively.
    launch_dates : dict[svn -> datetime] or None
        Used only to settle a contested position between two satellites that are
        both credibly at it: the earlier-launched one is taken to be the occupant
        and the later one a replacement waiting to take over. Omit for pure
        geometry.

    Returns
    -------
    dict with keys:
        plane_num, expandable_slot,
        baseline: list[dict]        # up to 4, one per matched baseline slot
        expansion: list[dict]       # up to 2, one per matched expansion slot
        unclassified: list[dict]    # USABLE sats not within any tolerance
        baseline_vacancies: list[str]
        expansion_vacancies: list[str]
        filled_positions_ordered: list[(label, expected_angle)]  # angular order
        effective_nominal_spacings: list[float]  # consecutive spacings, circular
        assigned_by_svn: dict[svn -> record]
    """
    expected_positions = GPS_EXPECTED_POSITIONS_BY_PLANE.get(plane_num) or {}
    expandable = GPS_EXPANDABLE_BASELINE_SLOT.get(plane_num)
    offsets = GPS_EXPANSION_OFFSETS_BY_PLANE.get(plane_num) or {}

    baseline_labels = set(GPS_BASELINE_SLOT_ORDER_BY_PLANE.get(plane_num) or [])
    expansion_label_offset = {}
    if expandable and offsets:
        expansion_label_offset[expandable + 'A'] = offsets.get('aft', 0)
        expansion_label_offset[expandable + 'F'] = offsets.get('fore', 0)

    empty_result = {
        'plane_num': plane_num,
        'expandable_slot': expandable,
        'baseline': [],
        'expansion': [],
        'unclassified': [],
        'baseline_vacancies': sorted(baseline_labels),
        'expansion_vacancies': sorted(expansion_label_offset.keys()),
        'filled_positions_ordered': [],
        'effective_nominal_spacings': [],
        'assigned_by_svn': {},
    }
    if not expected_positions or plot_df is None or plot_df.empty:
        return empty_result

    usable_df = plot_df[plot_df['Status'] == 'USABLE'].copy()
    if usable_df.empty:
        return empty_result

    sats = [
        {
            'svn': row['SVN'],
            'prn': row.get('PRN', ''),
            'raw_slot': row.get('Slot', 'Unk'),
            'angle': float(row['Plot_Angle']) % 360,
        }
        for _, row in usable_df.iterrows()
    ]

    pos_items = list(expected_positions.items())  # [(label, angle), ...]

    # Build all candidate (sat_idx, pos_idx, |dev|, signed_dev) pairs within
    # the per-kind tolerance. Then run greedy assignment in ascending |dev|.
    candidates = []
    for si, sat in enumerate(sats):
        for pi, (label, pos_angle) in enumerate(pos_items):
            signed = circular_delta_degrees(sat['angle'], pos_angle)
            adev = abs(signed)
            tol = expansion_tol if label in expansion_label_offset else baseline_tol
            if adev <= tol:
                candidates.append((adev, signed, si, pi))
    candidates.sort(key=lambda t: t[0])

    assigned_sat = {}
    assigned_pos = {}
    for adev, signed, si, pi in candidates:
        if si in assigned_sat or pi in assigned_pos:
            continue
        assigned_sat[si] = (pi, signed, adev)
        assigned_pos[pi] = si

    # Nearest-wins cannot resolve a co-located pair: on live plane B, PRN 12
    # (2006) sits 2.64 deg from B04 and PRN 13 (2026) 2.22 deg from it, so the
    # slot went to the newcomer and the incumbent was demoted to auxiliary. When
    # both are credibly at the position, launch date decides -- the incumbent
    # holds the slot, the successor shadows it.
    if launch_dates:
        _resolve_gps_occupant_ties(
            sats, pos_items, assigned_sat, assigned_pos, launch_dates,
            expansion_label_offset, baseline_tol, expansion_tol
        )

    baseline_records = []
    expansion_records = []
    assigned_by_svn = {}
    for si, sat in enumerate(sats):
        if si not in assigned_sat:
            continue
        pi, signed, adev = assigned_sat[si]
        label, pos_angle = pos_items[pi]
        record = {
            'svn': sat['svn'],
            'prn': sat['prn'],
            'raw_slot': sat['raw_slot'],
            'assigned_label': label,
            'expected_angle': pos_angle,
            'actual_angle': sat['angle'],
            'deviation': round(signed, 3),
        }
        if label in expansion_label_offset:
            expected_offset = expansion_label_offset[label]
            record['position'] = 'aft' if label.endswith('A') else 'fore'
            record['expected_offset'] = expected_offset
            record['actual_offset'] = round(expected_offset + signed, 3)
            expansion_records.append(record)
        else:
            baseline_records.append(record)
        assigned_by_svn[sat['svn']] = record

    unclassified_records = []
    for si, sat in enumerate(sats):
        if si in assigned_sat:
            continue
        nearest_pi = min(
            range(len(pos_items)),
            key=lambda p: abs(circular_delta_degrees(sat['angle'], pos_items[p][1])),
        )
        nearest_label, nearest_angle = pos_items[nearest_pi]
        unclassified_records.append({
            'svn': sat['svn'],
            'prn': sat['prn'],
            'raw_slot': sat['raw_slot'],
            'actual_angle': sat['angle'],
            'nearest_label': nearest_label,
            'nearest_expected_angle': nearest_angle,
            'nearest_deviation': round(circular_delta_degrees(sat['angle'], nearest_angle), 3),
        })

    filled_labels = {r['assigned_label'] for r in baseline_records + expansion_records}
    baseline_vacancies = sorted(baseline_labels - filled_labels)
    expansion_vacancies = sorted(set(expansion_label_offset.keys()) - filled_labels)

    filled_ordered = sorted(
        ((r['assigned_label'], r['expected_angle']) for r in baseline_records + expansion_records),
        key=lambda pair: pair[1],
    )
    effective_spacings = []
    if filled_ordered:
        angles = [angle for _, angle in filled_ordered]
        for i in range(len(angles)):
            nxt = angles[(i + 1) % len(angles)]
            effective_spacings.append((nxt - angles[i]) % 360)

    return {
        'plane_num': plane_num,
        'expandable_slot': expandable,
        'baseline': baseline_records,
        'expansion': expansion_records,
        'unclassified': unclassified_records,
        'baseline_vacancies': baseline_vacancies,
        'expansion_vacancies': expansion_vacancies,
        'filled_positions_ordered': filled_ordered,
        'effective_nominal_spacings': effective_spacings,
        'assigned_by_svn': assigned_by_svn,
    }


LEGACY_FRAME_COLUMNS = [
    'Key', 'Name', 'NORAD', 'PRN', 'SATCAT_Status', 'Launch_Date', 'Tier',
    'RAAN_Offset', 'Argument_Latitude', 'Relative_Argument_Latitude',
    'Plot_Angle', 'Nearest_Slot', 'Slot_Offset', 'TLE_Epoch',
]


def legacy_slot_proximity(plot_angle_deg, plane_num):
    """Nearest expected slot position in a GPS plane, and the signed offset to it.

    Takes a Plot_Angle, not a raw argument of latitude: the expected-position
    table is expressed in the plane's fitted frame, which is the same frame
    build_gps_plane_occupancy compares against.

    Returns (label, signed_offset_deg), or (None, None) for an unknown plane.
    """
    positions = GPS_EXPECTED_POSITIONS_BY_PLANE.get(plane_num) or {}
    if not positions:
        return None, None

    best_label = None
    best_offset = None
    for label, angle in positions.items():
        offset = circular_delta_degrees(plot_angle_deg, angle)
        if best_offset is None or abs(offset) < abs(best_offset):
            best_label = label
            best_offset = offset
    return best_label, best_offset


def build_legacy_plane_frame(legacy_states, legacy_assignments, plane_num,
                             ref_argument_latitude, plot_angle_offset):
    """Build the per-plane display frame for legacy satellites resident in a plane.

    Applies the identical angle transform active satellites receive (see
    app2.py:4721-4722) so legacy markers land in the same frame as the plane's
    real satellites and the nominal slot positions.

    Departed satellites are excluded here rather than filtered downstream: their
    plane is None by construction, so a >15 deg drifter can never reach a polar
    chart that assumes coplanarity.
    """
    rows = []
    for key, assignment in legacy_assignments.items():
        if assignment.get('plane') != plane_num:
            continue
        state = legacy_states.get(key) or {}
        argument_latitude = state.get('argument_latitude')
        if argument_latitude is None:
            continue

        relative = plane_relative_argument_latitude(argument_latitude, ref_argument_latitude)
        plot_angle = plane_plot_angle(relative, plot_angle_offset)
        nearest_slot, slot_offset = legacy_slot_proximity(plot_angle, plane_num)

        roster_entry = GPS_LEGACY_SATS.get(key, {})
        rows.append({
            'Key': key,
            'Name': roster_entry.get('name', key),
            'NORAD': roster_entry.get('norad', 0),
            'PRN': roster_entry.get('prn', ''),
            'SATCAT_Status': roster_entry.get('satcat_status', 'unknown'),
            'Launch_Date': roster_entry.get('launch_date', ''),
            'Tier': assignment.get('tier', GPS_LEGACY_TIER_HOLDING),
            'RAAN_Offset': assignment.get('raan_offset_deg'),
            'Argument_Latitude': argument_latitude,
            'Relative_Argument_Latitude': relative,
            'Plot_Angle': plot_angle,
            'Nearest_Slot': nearest_slot,
            'Slot_Offset': slot_offset,
            'TLE_Epoch': state.get('tle_epoch'),
        })

    return pd.DataFrame(rows, columns=LEGACY_FRAME_COLUMNS)


# An auxiliary parked within this angle of an occupied position is read as a
# replacement shadowing that position, rather than as an unexplained extra.
# Half the smallest nominal in-plane spacing (~30.1 deg): beyond this a satellite
# is nearer the midpoint between two slots than to either, so "parked beside the
# incumbent" stops being a fair description of it.
GPS_AUXILIARY_SHADOW_DEG = 15.0

# Display vocabulary for a GPS slot's configuration.
GPS_SLOT_STATE_LABELS = {
    'baseline': 'baseline',
    'expanded': 'expanded (fore+aft)',
    'transitional': 'transitional (centre + expansion)',
    'partial': 'PARTIAL - lost one of the expanded pair',
    'vacant': 'vacant',
}


def summarize_gps_plane_slots(occupancy):
    """
    Collapse a plane's occupancy map onto the four SPS PS slots, and explain the
    satellites that are not in one.

    Slot occupancy follows SPS PS A.7.2.4: a slot is occupied when either
      (1) its baseline centre from Table 3.2-1 carries a satellite, or
      (2) both expanded positions from Table 3.2-2 carry one.
    A slot holding only one half of its expanded pair has *lost* availability
    (3.7.1 note 3), so it does not count as occupied.

    This matters for scoring. Counting "satellites at baseline centres" punishes
    a plane for expanding a slot -- an expanded plane has only three centre
    occupants but all four slots filled, and expansion is defined as
    equivalent-or-better, never worse.

    On auxiliaries: SPS PS 3.2 says auxiliary satellites "occupy other locations
    in the orbital planes" and that "there are no a priori specified slots" for
    them. So GPS auxiliaries are real but position-less. Operationally they are
    replacements parked near the satellite they are due to take over from, so
    each is reported with the position it shadows.

    Returns
    -------
    dict with keys:
        slot_states       {baseline_slot_label: state}
        slots_occupied    int, 0-4
        slots_total       int
        expanded_slots    [label]  slots currently in the expanded configuration
        degraded_slots    [label]  slots that lost half of an expanded pair
        auxiliaries       [{'svn','prn','angle','shadows','offset_deg'}]
    """
    plane_num = occupancy.get('plane_num')
    baseline_labels = list(GPS_BASELINE_SLOT_ORDER_BY_PLANE.get(plane_num) or [])
    expandable = occupancy.get('expandable_slot')

    filled = {record['assigned_label'] for record in occupancy.get('baseline', [])}
    filled |= {record['assigned_label'] for record in occupancy.get('expansion', [])}

    slot_states = {}
    for label in baseline_labels:
        centre = label in filled
        if label != expandable:
            slot_states[label] = 'baseline' if centre else 'vacant'
            continue

        aft = f'{label}A' in filled
        fore = f'{label}F' in filled
        if centre and (aft or fore):
            slot_states[label] = 'transitional'
        elif centre:
            slot_states[label] = 'baseline'
        elif aft and fore:
            slot_states[label] = 'expanded'
        elif aft or fore:
            slot_states[label] = 'partial'
        else:
            slot_states[label] = 'vacant'

    occupied_states = {'baseline', 'expanded', 'transitional'}
    slots_occupied = sum(1 for state in slot_states.values() if state in occupied_states)

    occupied_positions = [
        (record['assigned_label'], record['actual_angle'])
        for record in occupancy.get('baseline', []) + occupancy.get('expansion', [])
    ]
    auxiliaries = []
    for record in occupancy.get('unclassified', []):
        shadows, offset = None, None
        if occupied_positions:
            label, angle = min(
                occupied_positions,
                key=lambda item: circular_distance_degrees(record['actual_angle'], item[1]),
            )
            separation = circular_delta_degrees(record['actual_angle'], angle)
            if abs(separation) <= GPS_AUXILIARY_SHADOW_DEG:
                shadows, offset = label, round(separation, 2)
        auxiliaries.append({
            'svn': record['svn'],
            'prn': record['prn'],
            'angle': record['actual_angle'],
            'shadows': shadows,
            'offset_deg': offset,
        })

    return {
        'slot_states': slot_states,
        'slots_occupied': slots_occupied,
        'slots_total': len(baseline_labels),
        'expanded_slots': [l for l, s in slot_states.items() if s == 'expanded'],
        'degraded_slots': [l for l, s in slot_states.items() if s == 'partial'],
        'auxiliaries': auxiliaries,
    }


# Angular tolerance for matching an Auxiliary/Spare satellite to a vacant
# baseline slot: half of the fixed 45 deg baseline spacing used by Galileo,
# BeiDou, and GLONASS. GPS has its own tolerance constants (see
# GPS_BASELINE_CLASSIFY_TOL / GPS_EXPANSION_CLASSIFY_TOL above) and does not
# use this constant.
BASELINE_FILL_CLASSIFY_TOL = 15.0


def build_baseline_expected_positions(slot_angles, regular_slot_count):
    """
    {label: angle} for baseline slots only, e.g. Galileo's A01..A08 grid,
    drawn from a constellation's slot_angles config. Excludes auxiliary-slot
    entries such as A17 whose slot number exceeds regular_slot_count.
    """
    positions = {}
    for label, angle in slot_angles.items():
        try:
            slot_num = int(str(label)[1:])
        except (ValueError, IndexError):
            continue
        if slot_num <= regular_slot_count:
            positions[label] = angle
    return positions


def build_baseline_fill_occupancy(plot_df, expected_positions, tol=BASELINE_FILL_CLASSIFY_TOL):
    """
    For a single plane's satellites (plot_df must carry 'SVN', 'Slot', 'Type',
    'Status', 'Plot_Angle'), detect Auxiliary/Spare satellites sitting within
    `tol` degrees of a baseline slot that has no USABLE Regular-by-name
    satellite. A filler only matches a vacancy in its own plane (the leading
    letter of its 'Slot', e.g. 'A' from 'A17') — expected_positions may cover
    multiple planes, and every plane reuses the same 0/45/.../315 angle set,
    so cross-plane matches would otherwise be indistinguishable from real
    ones. Greedy minimum-deviation assignment, one filler per vacant slot.

    Returns dict[svn -> {'assigned_label', 'expected_angle', 'actual_angle', 'deviation'}]
    """
    if plot_df is None or plot_df.empty or not expected_positions:
        return {}

    usable = plot_df[plot_df['Status'] == 'USABLE']
    occupied_labels = set(usable.loc[usable['Type'] == 'Regular', 'Slot'])
    vacancies = [(label, angle) for label, angle in expected_positions.items() if label not in occupied_labels]
    if not vacancies:
        return {}

    fillers = usable[usable['Type'].isin(['Auxiliary', 'Spare'])]
    if fillers.empty:
        return {}

    candidates = []
    for _, row in fillers.iterrows():
        # Extract plane from filler's Slot (e.g., 'A17' -> plane 'A'). Spares
        # commonly carry Slot == 'Unk' (see get_satellite_type); their plane
        # is unknowable from the label alone, so leave the constraint off and
        # fall back to pure tolerance-based matching for them.
        raw_slot = str(row['Slot']) if pd.notna(row['Slot']) else ''
        filler_plane = raw_slot[0] if raw_slot and raw_slot != 'Unk' else None
        actual_angle = float(row['Plot_Angle']) % 360
        for label, expected_angle in vacancies:
            # Only constrain to the same plane when the filler's plane is known.
            vacancy_plane = str(label)[0] if label else None
            if filler_plane is not None and filler_plane != vacancy_plane:
                continue
            signed = circular_delta_degrees(actual_angle, expected_angle)
            adev = abs(signed)
            if adev <= tol:
                candidates.append((adev, row['SVN'], label, expected_angle, actual_angle, signed))
    candidates.sort(key=lambda t: t[0])

    assigned_svn = set()
    assigned_label = set()
    result = {}
    for adev, svn, label, expected_angle, actual_angle, signed in candidates:
        if svn in assigned_svn or label in assigned_label:
            continue
        assigned_svn.add(svn)
        assigned_label.add(label)
        result[svn] = {
            'assigned_label': label,
            'expected_angle': expected_angle,
            'actual_angle': actual_angle,
            'deviation': round(signed, 3),
        }
    return result


def apply_baseline_fill_labels(plot_df, slot_angles, regular_slot_count, constellation_name,
                                tol=BASELINE_FILL_CLASSIFY_TOL):
    """
    Add 'Baseline_Fill_Label' and 'Baseline_Fill_Deviation' columns to a
    per-plane polar-chart frame (see build_baseline_fill_occupancy for the
    matching rules). No-op for GPS, which has its own richer expansion-slot
    occupancy model (see build_gps_plane_occupancy).
    """
    result = plot_df.copy()
    result['Baseline_Fill_Label'] = None
    result['Baseline_Fill_Deviation'] = np.nan
    if constellation_name == 'GPS' or result.empty:
        return result
    expected_positions = build_baseline_expected_positions(slot_angles, regular_slot_count)
    fill_occupancy = build_baseline_fill_occupancy(result, expected_positions, tol=tol)
    for svn, record in fill_occupancy.items():
        mask = result['SVN'] == svn
        result.loc[mask, 'Baseline_Fill_Label'] = record['assigned_label']
        result.loc[mask, 'Baseline_Fill_Deviation'] = record['deviation']
    return result


@st.cache_data
def parse_tle_line(line1, line2):
    """Parse TLE lines and extract orbital parameters"""
    try:
        # Line 1: Epoch
        epoch_year = int(line1[18:20])
        epoch_year = 2000 + epoch_year if epoch_year < 57 else 1900 + epoch_year
        epoch_day = float(line1[20:32])
        
        # Line 2: Inclination (9-16), RAAN (18-25), Eccentricity (27-33)
        inclination = float(line2[8:16].strip())
        raan = float(line2[17:25].strip())
        eccentricity = float('0.' + line2[26:33].strip())
        arg_perigee = float(line2[34:42].strip())
        mean_anomaly = float(line2[43:51].strip())
        mean_motion = float(line2[52:63].strip())
        
        # Calculate epoch datetime
        epoch = datetime(epoch_year, 1, 1, tzinfo=UTC) + timedelta(days=epoch_day - 1)
        
        return {
            'inclination': inclination,
            'raan': raan,
            'eccentricity': eccentricity,
            'arg_perigee': arg_perigee,
            'mean_anomaly': mean_anomaly,
            'mean_motion': mean_motion,
            'epoch': epoch
        }
    except Exception as e:
        st.error(f"Error parsing TLE: {e}")
        return None


def calculate_satellite_age_score(launch_date, design_life_years=12):
    """
    Calculate age-based health score for a satellite.
    
    Parameters
    -----------
    launch_date : datetime
        Launch date of the satellite
    design_life_years : int
        Design life in years (default 12 for Galileo, 10 for BeiDou)
        
    Returns
    --------
    float
        Score from 0-100 (100 = brand new, 0 = way past design life)
    """
    if launch_date is None:
        return 50.0  # Neutral score if no data
    
    # Keep age-based health pinned to the same epoch as the orbital geometry.
    age_reference = STATIC_ANALYSIS_EPOCH
    if age_reference is None:
        raise ValueError("Static analysis epoch must be loaded before age scoring.")
    launch_date = launch_date.replace(tzinfo=UTC) if launch_date.tzinfo is None else launch_date.astimezone(UTC)
    age_years = (age_reference - launch_date).days / 365.25
    
    # Score calculation:
    # 0-50% of design life: 100 points (excellent)
    # 50-80% of design life: 90-100 points (good)
    # 80-100% of design life: 70-90 points (fair)
    # 100-120% of design life: 40-70 points (concerning)
    # >120% of design life: 0-40 points (critical)
    
    age_ratio = age_years / design_life_years
    
    if age_ratio <= 0.5:
        return 100.0
    elif age_ratio <= 0.8:
        return 100 - (age_ratio - 0.5) * (10 / 0.3)  # 100 to 90
    elif age_ratio <= 1.0:
        return 90 - (age_ratio - 0.8) * (20 / 0.2)  # 90 to 70
    elif age_ratio <= 1.2:
        return 70 - (age_ratio - 1.0) * (30 / 0.2)  # 70 to 40
    else:
        return max(0, 40 - (age_ratio - 1.2) * 20)  # 40 to 0


def calculate_plane_age_health(sat_age_entries, baseline_n):
    """
    Plane age health: youngest-N base + spare bonus.

    Old satellites retained beyond the baseline slot count no longer penalize
    the plane; usable spares still within design life earn bonus credit.

    Parameters
    ----------
    sat_age_entries : list of dict
        [{'svn': str, 'age_ratio': float, 'score': float}, ...] for usable
        satellites with known launch dates. 'score' is the
        calculate_satellite_age_score value, 'age_ratio' = age / design life.
    baseline_n : int
        Baseline slot count for the constellation (GPS 4, others 8).

    Returns
    --------
    dict with:
        age_base_points   : 0-20, (mean score of youngest N) / 100 * 20
        spare_bonus_points: 0-5, +2.5 per extra sat with age_ratio < 1.0
        age_points        : base + bonus (0-25)
        youngest_mean     : mean score of youngest N on the 0-100 scale
        counted_by_svn    : {svn: 'scored' | 'bonus' | 'not_counted'}
    """
    if not sat_age_entries:
        return {'age_base_points': 12.5, 'spare_bonus_points': 0.0,
                'age_points': 12.5, 'youngest_mean': 50.0, 'counted_by_svn': {}}

    ordered = sorted(sat_age_entries, key=lambda entry: entry['age_ratio'])
    youngest = ordered[:baseline_n]
    rest = ordered[baseline_n:]

    youngest_mean = float(np.mean([entry['score'] for entry in youngest]))
    age_base_points = (youngest_mean / 100.0) * 20.0

    counted_by_svn = {entry['svn']: 'scored' for entry in youngest}
    spare_bonus_points = 0.0
    for entry in rest:
        if entry['age_ratio'] < 1.0 and spare_bonus_points < 5.0:
            spare_bonus_points += 2.5
            counted_by_svn[entry['svn']] = 'bonus'
        else:
            counted_by_svn[entry['svn']] = 'not_counted'

    spare_bonus_points = min(5.0, spare_bonus_points)
    return {'age_base_points': age_base_points,
            'spare_bonus_points': spare_bonus_points,
            'age_points': age_base_points + spare_bonus_points,
            'youngest_mean': youngest_mean,
            'counted_by_svn': counted_by_svn}


def filter_satellites_by_epoch(parsed_data, reference_svn, max_days_diff=3, min_satellites=None):
    """
    Filter satellites to include those within max_days_diff days of the reference satellite's epoch.
    Uses adaptive fallback to ensure minimum satellite coverage.
    """
    if reference_svn not in parsed_data:
        return parsed_data, {'excluded_count': 0, 'expanded_window': False, 'epoch_warnings': {}}
    
    ref_epoch = parsed_data[reference_svn]['epoch']
    
    # Try strict filtering first
    filtered_data = {}
    epoch_warnings = {}
    
    for svn, params in parsed_data.items():
        sat_epoch = params['epoch']
        days_diff = abs((sat_epoch - ref_epoch).total_seconds() / 86400)
        
        if days_diff <= max_days_diff:
            filtered_data[svn] = params
            epoch_warnings[svn] = {'days_diff': days_diff, 'quality': 'good'}
        else:
            epoch_warnings[svn] = {'days_diff': days_diff, 'quality': 'excluded'}
    
    # Adaptive fallback if too few satellites
    expanded_window = False
    if min_satellites and len(filtered_data) < min_satellites:
        # Expand window progressively: 3 -> 7 -> 14 -> 30 days
        for fallback_window in [7, 14, 30]:
            filtered_data = {}
            for svn, params in parsed_data.items():
                sat_epoch = params['epoch']
                days_diff = abs((sat_epoch - ref_epoch).total_seconds() / 86400)
                
                if days_diff <= fallback_window:
                    filtered_data[svn] = params
                    # Mark quality based on deviation
                    if days_diff <= 3:
                        epoch_warnings[svn] = {'days_diff': days_diff, 'quality': 'good'}
                    elif days_diff <= 7:
                        epoch_warnings[svn] = {'days_diff': days_diff, 'quality': 'fair'}
                    else:
                        epoch_warnings[svn] = {'days_diff': days_diff, 'quality': 'poor'}
                else:
                    epoch_warnings[svn] = {'days_diff': days_diff, 'quality': 'excluded'}
            
            if len(filtered_data) >= min_satellites:
                expanded_window = fallback_window
                break
        
        # If still not enough, include all satellites
        if len(filtered_data) < min_satellites:
            filtered_data = parsed_data.copy()
            expanded_window = 999
            for svn, params in parsed_data.items():
                sat_epoch = params['epoch']
                days_diff = abs((sat_epoch - ref_epoch).total_seconds() / 86400)
                if days_diff <= 3:
                    epoch_warnings[svn] = {'days_diff': days_diff, 'quality': 'good'}
                elif days_diff <= 7:
                    epoch_warnings[svn] = {'days_diff': days_diff, 'quality': 'fair'}
                elif days_diff <= 14:
                    epoch_warnings[svn] = {'days_diff': days_diff, 'quality': 'poor'}
                else:
                    epoch_warnings[svn] = {'days_diff': days_diff, 'quality': 'very_poor'}
    
    excluded_count = len(parsed_data) - len(filtered_data)
    
    metadata = {
        'excluded_count': excluded_count,
        'expanded_window': expanded_window,
        'epoch_warnings': epoch_warnings
    }
    
    return filtered_data, metadata

def get_skyfield_timescale():
    """Create Skyfield's timescale lazily so imports stay optional."""
    global _SKYFIELD_TIMESCALE
    if _SKYFIELD_TIMESCALE is None:
        _SKYFIELD_TIMESCALE = skyfield_load.timescale(builtin=True)
    return _SKYFIELD_TIMESCALE


def ensure_utc_datetime(value):
    """Return a timezone-aware UTC datetime."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_tle_record(svn, tle):
    """Parse a fetched TLE/OMM record and retain parameters needed by Skyfield."""
    if 'omm_data' in tle:
        return tle
        
    params = parse_tle_line(tle.get('line1', ''), tle.get('line2', ''))
    if not params:
        return None

    params['tle_line1'] = tle['line1']
    params['tle_line2'] = tle['line2']
    params['tle_name'] = tle.get('line0', svn)
    return params


def derive_orbital_angles_from_state(position_km, velocity_km_s):
    """Derive osculating RAAN, inclination, and argument of latitude from r/v."""
    position = np.array(position_km, dtype=float)
    velocity = np.array(velocity_km_s, dtype=float)
    angular_momentum = np.cross(position, velocity)

    r_norm = np.linalg.norm(position)
    h_norm = np.linalg.norm(angular_momentum)
    if r_norm < 1e-12 or h_norm < 1e-12:
        return None

    inclination = np.degrees(np.arccos(np.clip(angular_momentum[2] / h_norm, -1.0, 1.0)))
    node_vector = np.cross(np.array([0.0, 0.0, 1.0]), angular_momentum)
    node_norm = np.linalg.norm(node_vector)

    if node_norm < 1e-12:
        raan = 0.0
        argument_latitude = np.degrees(np.arctan2(position[1], position[0])) % 360
    else:
        raan = np.degrees(np.arctan2(node_vector[1], node_vector[0])) % 360
        cos_u = np.dot(node_vector, position) / (node_norm * r_norm)
        sin_u = np.dot(angular_momentum, np.cross(node_vector, position)) / (h_norm * node_norm * r_norm)
        argument_latitude = np.degrees(np.arctan2(sin_u, np.clip(cos_u, -1.0, 1.0))) % 360

    return {
        'raan': raan,
        'inclination': inclination,
        'argument_latitude': argument_latitude,
        'radius_km': r_norm,
    }


def get_skyfield_ecef_position(geocentric):
    """Return an Earth-fixed position from a Skyfield geocentric object."""
    if hasattr(geocentric, 'itrf_xyz'):
        return np.array(geocentric.itrf_xyz().km, dtype=float)
    if skyfield_itrs is not None:
        return np.array(geocentric.frame_xyz(skyfield_itrs).km, dtype=float)
    return np.array(geocentric.position.km, dtype=float)


def propagate_tle_with_sgp4(params, analysis_epoch, fallback_name):
    """Propagate one TLE to the common analysis epoch and return derived elements."""
    if not SKYFIELD_AVAILABLE:
        raise RuntimeError("Skyfield is required for common-epoch SGP4 propagation.")

    analysis_epoch = ensure_utc_datetime(analysis_epoch)
    ts = get_skyfield_timescale()
    
    if 'omm_data' in params:
        satellite = EarthSatellite.from_omm(ts, params['omm_data'])
    else:
        line1 = params.get('tle_line1')
        line2 = params.get('tle_line2')
        if not line1 or not line2:
            raise ValueError("Missing raw TLE/OMM data for Skyfield propagation.")
        satellite = EarthSatellite(line1, line2, params.get('tle_name', fallback_name), ts)
        
    geocentric = satellite.at(ts.from_datetime(analysis_epoch))

    position_eci_km = np.array(geocentric.position.km, dtype=float)
    velocity_eci_km_s = np.array(geocentric.velocity.km_per_s, dtype=float)
    derived_angles = derive_orbital_angles_from_state(position_eci_km, velocity_eci_km_s)
    if derived_angles is None:
        raise ValueError("Could not derive orbital geometry from propagated state vector.")

    propagated = dict(params)
    propagated.update({
        'tle_epoch': params['epoch'],
        'analysis_epoch': analysis_epoch,
        'raan_tle_epoch': params['raan'],
        'inclination_tle_epoch': params['inclination'],
        'mean_anomaly_tle_epoch': params['mean_anomaly'],
        'arg_perigee_tle_epoch': params['arg_perigee'],
        'raan': derived_angles['raan'],
        'inclination': derived_angles['inclination'],
        'argument_latitude': derived_angles['argument_latitude'],
        'radius_km': derived_angles['radius_km'],
        'position_eci_km': tuple(position_eci_km),
        'velocity_eci_km_s': tuple(velocity_eci_km_s),
        'position_ecef_km': tuple(get_skyfield_ecef_position(geocentric)),
        'sgp4_propagated': True,
    })
    return propagated


def propagate_all_tles_with_sgp4(parsed_data, analysis_epoch):
    """Propagate all parsed TLE records to one shared epoch."""
    propagated_data = {}
    errors = []

    for svn, params in parsed_data.items():
        try:
            propagated_data[svn] = propagate_tle_with_sgp4(params, analysis_epoch, svn)
        except Exception as exc:
            errors.append((svn, str(exc)))

    return propagated_data, errors


def print_argument_of_latitude_debug(constellation_name, plane_num, plot_df, filtered_plane_data, ref_svn, analysis_epoch):
    """Print per-plane argument-of-latitude deltas and summary stats to the console."""
    if plot_df.empty or ref_svn not in filtered_plane_data:
        return

    if not SKYFIELD_AVAILABLE:
        print(
            f"\nARGUMENT OF LATITUDE DEBUG ({constellation_name}) - Plane {plane_num} skipped: "
            "Skyfield is not installed, so SGP4 propagation is unavailable."
        )
        return

    debug_rows = []
    for _, plot_row in plot_df.iterrows():
        if plot_row.get('Type') == 'Auxiliary':
            continue

        svn = plot_row['SVN']
        params = filtered_plane_data.get(svn)
        if not params:
            continue

        argument_latitude = params.get('argument_latitude')
        if argument_latitude is None:
            continue

        epoch_diff_days = (params['tle_epoch'] - analysis_epoch).total_seconds() / 86400
        debug_rows.append({
            'SVN': svn,
            'PRN': plot_row.get('PRN', ''),
            'Slot': plot_row.get('Slot', ''),
            'Type': plot_row.get('Type', ''),
            'Arg_Lat': argument_latitude,
            'Epoch_Diff_Days': epoch_diff_days,
        })

    if not debug_rows:
        return

    ref_row = next((row for row in debug_rows if row['SVN'] == ref_svn), debug_rows[0])
    ref_arg_lat = ref_row['Arg_Lat']

    for row in debug_rows:
        row['Delta_From_Ref'] = circular_delta_degrees(row['Arg_Lat'], ref_arg_lat)

    sorted_rows = sorted(debug_rows, key=lambda row: row['Arg_Lat'])
    if len(sorted_rows) > 1:
        for idx, row in enumerate(sorted_rows):
            next_row = sorted_rows[(idx + 1) % len(sorted_rows)]
            row['Next_Delta'] = (next_row['Arg_Lat'] - row['Arg_Lat']) % 360
        adjacent_spacings = [row['Next_Delta'] for row in sorted_rows]
    else:
        sorted_rows[0]['Next_Delta'] = 0.0
        adjacent_spacings = []

    pairwise_deltas = []
    for idx, row in enumerate(sorted_rows):
        for other_row in sorted_rows[idx + 1:]:
            pairwise_deltas.append(
                circular_distance_degrees(row['Arg_Lat'], other_row['Arg_Lat'])
            )

    angles_rad = np.radians([row['Arg_Lat'] for row in sorted_rows])
    mean_sin = np.mean(np.sin(angles_rad))
    mean_cos = np.mean(np.cos(angles_rad))
    circular_mean = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360
    resultant_length = max(np.sqrt(mean_sin ** 2 + mean_cos ** 2), 1e-12)
    circular_std = np.degrees(np.sqrt(max(0.0, -2 * np.log(resultant_length))))

    print("\n" + "=" * 100)
    print(f"ARGUMENT OF LATITUDE DEBUG ({constellation_name}) - Plane {plane_num}")
    print("=" * 100)
    print(
        "Using Skyfield/SGP4 propagated state vectors at the common analysis epoch. "
        "u is measured from the propagated ascending node to the propagated position vector."
    )
    print(f"Reference: {ref_svn} | Epoch: {analysis_epoch.strftime('%Y-%m-%d %H:%M:%S UTC')} | Ref u: {ref_arg_lat:.2f} deg")
    print(
        f"{'SVN':<15} {'PRN':<8} {'Slot':<8} {'Type':<10} "
        f"{'Epoch d':>9} {'u SGP4':>10} {'dRef':>10} {'dNext':>10}"
    )
    print("-" * 100)
    for row in sorted_rows:
        print(
            f"{row['SVN']:<15} {row['PRN']:<8} {row['Slot']:<8} {row['Type']:<10} "
            f"{row['Epoch_Diff_Days']:>9.2f} {row['Arg_Lat']:>10.2f} "
            f"{row['Delta_From_Ref']:>10.2f} "
            f"{row['Next_Delta']:>10.2f}"
        )

    print("Stats:")
    print(f"  Satellites: {len(sorted_rows)}")
    print(f"  Circular mean u: {circular_mean:.2f} deg")
    print(f"  Circular std u: {circular_std:.2f} deg")
    if adjacent_spacings:
        expected_spacing = 360 / len(sorted_rows)
        spacing_errors = [spacing - expected_spacing for spacing in adjacent_spacings]
        print(
            "  Adjacent spacing: "
            f"mean={np.mean(adjacent_spacings):.2f} deg, "
            f"std={np.std(adjacent_spacings):.2f} deg, "
            f"min={min(adjacent_spacings):.2f} deg, "
            f"max={max(adjacent_spacings):.2f} deg"
        )
        print(
            "  Uniform spacing error: "
            f"expected={expected_spacing:.2f} deg, "
            f"mean_abs={np.mean(np.abs(spacing_errors)):.2f} deg, "
            f"max_abs={np.max(np.abs(spacing_errors)):.2f} deg"
        )
    if pairwise_deltas:
        print(
            "  Pairwise separation: "
            f"mean={np.mean(pairwise_deltas):.2f} deg, "
            f"std={np.std(pairwise_deltas):.2f} deg, "
            f"min={min(pairwise_deltas):.2f} deg, "
            f"max={max(pairwise_deltas):.2f} deg"
        )
    print("=" * 100 + "\n")


def calculate_drift(tle_params, plane, plane_mean_raan=None,
                    nominal_mean_motion=GAL_NOMINAL_MEAN_MOTION,
                    nominal_inclination=GAL_NOMINAL_INCLINATION,
                    orbital_radius=GAL_ORBITAL_RADIUS,
                    km_per_degree=GAL_KM_PER_DEGREE,
                    nominal_raan_map=None):
    """Calculate in-plane and across-track drift for any GNSS constellation."""
    if nominal_raan_map is None:
        nominal_raan_map = GAL_NOMINAL_RAAN
    
    # In-plane drift calculation
    delta_n = tle_params['mean_motion'] - nominal_mean_motion
    drift_angle_per_day = 360.0 * delta_n  # degrees per day
    in_plane_drift_km_per_day = km_per_degree * drift_angle_per_day
    
    orbital_period = 24.0 / tle_params['mean_motion']  # hours
    
    # Across-track drift calculation
    # Use actual plane mean RAAN if provided, otherwise use nominal
    reference_raan = plane_mean_raan if plane_mean_raan is not None else nominal_raan_map.get(plane, 0)
    delta_raan = tle_params['raan'] - reference_raan
    
    # Normalize delta_raan to [-180, 180]
    if delta_raan > 180:
        delta_raan -= 360
    elif delta_raan < -180:
        delta_raan += 360
    
    delta_inclination = tle_params['inclination'] - nominal_inclination
    
    # Cross-track displacement from RAAN and inclination drift
    raan_cross_track_km = orbital_radius * np.radians(abs(delta_raan))
    inc_cross_track_km = orbital_radius * np.radians(abs(delta_inclination))
    cross_track_km = np.sqrt(raan_cross_track_km**2 + inc_cross_track_km**2)
    
    # Check if within tolerance
    in_plane_tolerance = abs(drift_angle_per_day) < 2.0  # degrees
    across_track_tolerance = abs(delta_raan) < 2.0 and abs(delta_inclination) < 2.0
    
    return {
        'delta_mean_motion': delta_n,
        'drift_angle_per_day': drift_angle_per_day,
        'in_plane_drift_km_day': in_plane_drift_km_per_day,
        'orbital_period_hours': orbital_period,
        'delta_raan': delta_raan,
        'delta_inclination': delta_inclination,
        'cross_track_km': cross_track_km,
        'in_plane_ok': in_plane_tolerance,
        'across_track_ok': across_track_tolerance,
        'mean_anomaly': tle_params['mean_anomaly']
    }

def calculate_satellite_position(tle_params, time_offset_seconds=0, frame='ecef'):
    """
    Return the common-epoch SGP4 position when available.
    """
    if frame == 'eci' and 'position_eci_km' in tle_params:
        return tuple(tle_params['position_eci_km'])
    if frame == 'ecef' and 'position_ecef_km' in tle_params:
        return tuple(tle_params['position_ecef_km'])
    raise ValueError(f"Missing common-epoch SGP4 {frame.upper()} position.")


def create_3d_constellation_globe(parsed_data, df, plane_mean_raans, active_sats,
                                  orbital_radius, nominal_inclination,
                                  constellation_name="Galileo",
                                  plane_numbers=None,
                                  legacy_states=None,
                                  legacy_assignments=None):
    """
    Create a 3D globe with orbital rings and current satellite positions.
    """
    earth_radius = 6378.137  # km
    plane_colors = {
        1: '#00B0FF',
        2: '#FFB300',
        3: '#66BB6A',
        4: '#EC407A',
        5: '#AB47BC',
        6: '#26A69A'
    }
    if plane_numbers is None:
        plane_numbers = sorted(plane_mean_raans.keys()) or [1, 2, 3]

    fig = go.Figure()

    lon = np.linspace(0, 2 * np.pi, 96)
    lat = np.linspace(-np.pi / 2, np.pi / 2, 48)
    earth_x = earth_radius * np.outer(np.cos(lon), np.cos(lat))
    earth_y = earth_radius * np.outer(np.sin(lon), np.cos(lat))
    earth_z = earth_radius * np.outer(np.ones_like(lon), np.sin(lat))
    surface_color = np.outer(np.ones_like(lon), np.sin(lat))

    fig.add_trace(go.Surface(
        x=earth_x,
        y=earth_y,
        z=earth_z,
        surfacecolor=surface_color,
        colorscale=[
            [0.00, '#082567'],
            [0.42, '#145DA0'],
            [0.50, '#2E7D32'],
            [0.58, '#145DA0'],
            [1.00, '#0A3D62']
        ],
        showscale=False,
        opacity=0.9,
        hoverinfo='skip',
        name='Earth'
    ))

    equator = np.linspace(0, 2 * np.pi, 241)
    fig.add_trace(go.Scatter3d(
        x=earth_radius * np.cos(equator),
        y=earth_radius * np.sin(equator),
        z=np.zeros_like(equator),
        mode='lines',
        line=dict(color='rgba(255,255,255,0.35)', width=2),
        name='Equator',
        hoverinfo='skip',
        showlegend=False
    ))

    def orbital_ring(raan_deg, inclination_deg, radius_km):
        theta = np.linspace(0, 2 * np.pi, 361)
        x_orb = radius_km * np.cos(theta)
        y_orb = radius_km * np.sin(theta)
        z_orb = np.zeros_like(theta)

        inc = np.radians(inclination_deg)
        raan = np.radians(raan_deg)

        x_inc = x_orb
        y_inc = y_orb * np.cos(inc) - z_orb * np.sin(inc)
        z_inc = y_orb * np.sin(inc) + z_orb * np.cos(inc)

        x = x_inc * np.cos(raan) - y_inc * np.sin(raan)
        y = x_inc * np.sin(raan) + y_inc * np.cos(raan)
        z = z_inc
        return x, y, z

    plotted_satellites = 0

    for plane_num in plane_numbers:
        plane_label = f"Plane {plane_num}"
        plane_df = df[df['Plane'] == plane_label].copy()
        if plane_df.empty:
            continue

        plane_color = plane_colors.get(plane_num, '#BDBDBD')
        plane_raan = plane_mean_raans.get(plane_num)
        if plane_raan is None:
            plane_raans = [
                parsed_data[svn]['raan']
                for svn in plane_df['SVN'].tolist()
                if svn in parsed_data
            ]
            plane_raan = float(np.mean(plane_raans)) if plane_raans else 0.0

        plane_inclinations = [
            parsed_data[svn]['inclination']
            for svn in plane_df['SVN'].tolist()
            if svn in parsed_data
        ]
        plane_radii = [
            parsed_data[svn].get('radius_km', orbital_radius)
            for svn in plane_df['SVN'].tolist()
            if svn in parsed_data
        ]
        plane_inclination = float(np.mean(plane_inclinations)) if plane_inclinations else nominal_inclination
        plane_radius = float(np.mean(plane_radii)) if plane_radii else orbital_radius

        ring_x, ring_y, ring_z = orbital_ring(plane_raan, plane_inclination, plane_radius)
        fig.add_trace(go.Scatter3d(
            x=ring_x,
            y=ring_y,
            z=ring_z,
            mode='lines',
            line=dict(color=plane_color, width=5),
            name=f"{plane_label} orbit",
            hoverinfo='skip'
        ))

        sat_x, sat_y, sat_z, labels, hover_rows = [], [], [], [], []
        for _, row in plane_df.iterrows():
            svn = row['SVN']
            if svn not in parsed_data:
                continue

            sat_info = active_sats.get(svn, {})
            x, y, z = calculate_satellite_position(parsed_data[svn], frame='eci')
            sat_x.append(x)
            sat_y.append(y)
            sat_z.append(z)
            labels.append(sat_info.get('slot') or sat_info.get('prn') or svn)
            hover_rows.append([
                svn,
                sat_info.get('prn', 'N/A'),
                sat_info.get('slot', 'N/A'),
                row.get('Type', 'N/A'),
                parsed_data[svn]['raan'],
                parsed_data[svn].get('argument_latitude', 0.0),
                sat_info.get('status', 'USABLE')
            ])

        if not sat_x:
            continue

        plotted_satellites += len(sat_x)
        fig.add_trace(go.Scatter3d(
            x=sat_x,
            y=sat_y,
            z=sat_z,
            mode='markers+text',
            marker=dict(
                size=5,
                color=plane_color,
                line=dict(color='white', width=1)
            ),
            text=labels,
            textposition='top center',
            textfont=dict(size=9, color='white'),
            customdata=hover_rows,
            name=f"{plane_label} satellites ({len(sat_x)})",
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "PRN: %{customdata[1]}<br>"
                "Slot: %{customdata[2]}<br>"
                f"Plane: {plane_num}<br>"
                "Type: %{customdata[3]}<br>"
                "RAAN: %{customdata[4]:.2f} deg<br>"
                "Argument latitude: %{customdata[5]:.2f} deg<br>"
                "Status: %{customdata[6]}<extra></extra>"
            )
        ))

    propagated_radii = [
        params.get('radius_km', orbital_radius)
        for params in parsed_data.values()
    ]
    max_extent = (max(propagated_radii) if propagated_radii else orbital_radius) * 1.08
    analysis_epochs = [
        params.get('analysis_epoch')
        for params in parsed_data.values()
        if params.get('analysis_epoch') is not None
    ]
    epoch_label = (
        f" at {analysis_epochs[0].strftime('%Y-%m-%d %H:%M:%S UTC')}"
        if analysis_epochs else ""
    )
    title_suffix = f"{plotted_satellites} satellites across active orbital planes{epoch_label}"

    # Legacy (retired / standby) satellites: dimmed, in their plane's colour, in
    # a separate trace so they read as clutter in the shell rather than as
    # constellation members.
    if legacy_states and legacy_assignments:
        legacy_x, legacy_y, legacy_z, legacy_text, legacy_colors = [], [], [], [], []
        for key, assignment in legacy_assignments.items():
            plane_num = assignment.get('plane')
            state = legacy_states.get(key) or {}
            position = state.get('position_eci_km')
            if plane_num is None or not position:
                continue
            legacy_x.append(position[0])
            legacy_y.append(position[1])
            legacy_z.append(position[2])
            legacy_colors.append(plane_colors.get(plane_num, '#9E9E9E'))
            legacy_text.append(
                f"{key} (legacy)<br>Plane {plane_num}<br>{assignment['tier']}"
            )

        if legacy_x:
            fig.add_trace(go.Scatter3d(
                x=legacy_x, y=legacy_y, z=legacy_z,
                mode='markers',
                marker=dict(size=4, color=legacy_colors,
                            symbol='diamond-open', opacity=0.45),
                text=legacy_text,
                name='Legacy (retired / standby)',
                hovertemplate='%{text}<extra></extra>'
            ))

    fig.update_layout(
        title={
            'text': f"{constellation_name} 3D Globe View<br><sub>{title_suffix}</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        scene=dict(
            bgcolor='#07111F',
            xaxis=dict(visible=False, range=[-max_extent, max_extent]),
            yaxis=dict(visible=False, range=[-max_extent, max_extent]),
            zaxis=dict(visible=False, range=[-max_extent, max_extent]),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.45, y=1.35, z=0.9))
        ),
        height=760,
        margin=dict(l=0, r=0, t=90, b=80),
        paper_bgcolor='#07111F',
        plot_bgcolor='#07111F',
        font=dict(color='white'),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(7,17,31,0.85)',
            bordercolor='rgba(255,255,255,0.18)',
            borderwidth=1
        ),
        hovermode='closest'
    )

    return fig


def ecef_to_enu(sat_ecef, observer_lat, observer_lon, observer_alt=0):
    """
    Convert ECEF coordinates to ENU (East-North-Up) coordinates relative to observer.
    """
    x_sat, y_sat, z_sat = sat_ecef
    
    # Convert observer position to ECEF
    lat_rad = np.radians(observer_lat)
    lon_rad = np.radians(observer_lon)
    
    # WGS84 parameters
    a = 6378.137  # Semi-major axis (km)
    f = 1/298.257223563  # Flattening
    e2 = 2*f - f**2  # Eccentricity squared
    
    # Radius of curvature
    N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
    
    # Observer ECEF position
    x_obs = (N + observer_alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y_obs = (N + observer_alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z_obs = (N * (1 - e2) + observer_alt) * np.sin(lat_rad)
    
    # Difference vector
    dx = x_sat - x_obs
    dy = y_sat - y_obs
    dz = z_sat - z_obs
    
    # Rotation to ENU
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    
    return east, north, up


def calculate_azimuth_elevation(sat_ecef, observer_lat, observer_lon, observer_alt=0):
    """
    Calculate azimuth and elevation angles to satellite from observer position.
    """
    east, north, up = ecef_to_enu(sat_ecef, observer_lat, observer_lon, observer_alt)
    
    # Calculate range
    range_km = np.sqrt(east**2 + north**2 + up**2)
    
    # Calculate azimuth (0° = North, 90° = East)
    azimuth = np.degrees(np.arctan2(east, north))
    if azimuth < 0:
        azimuth += 360
    
    # Calculate elevation
    elevation = np.degrees(np.arcsin(up / range_km))
    
    return azimuth, elevation, range_km


def ecef_to_geodetic(ecef):
    """Convert ECEF coordinates in kilometers to geodetic lat/lon/alt."""
    x, y, z = [float(value) for value in ecef]

    a = 6378.137
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = 2 * f - f**2
    ep2 = (a**2 - b**2) / b**2

    p = np.sqrt(x**2 + y**2)
    if p < 1e-12:
        lat = np.pi / 2 if z >= 0 else -np.pi / 2
        lon = 0.0
        alt = abs(z) - b
    else:
        lon = np.arctan2(y, x)
        theta = np.arctan2(z * a, p * b)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        lat = np.arctan2(
            z + ep2 * b * sin_theta**3,
            p - e2 * a * cos_theta**3
        )
        n = a / np.sqrt(1 - e2 * np.sin(lat)**2)
        alt = p / np.cos(lat) - n

    lat_deg = np.degrees(lat)
    lon_deg = ((np.degrees(lon) + 540) % 360) - 180
    return lat_deg, lon_deg, alt


def get_dop_satellite_label(svn, sats_lookup):
    """Return a compact satellite label for DOP tables and maps."""
    if sats_lookup and svn in sats_lookup:
        return sats_lookup[svn].get('prn') or svn
    return svn


def get_visibility_color(elevation):
    """Color visible satellites by elevation band."""
    if elevation >= 60:
        return '#2E7D32'
    if elevation >= 30:
        return '#F9A825'
    return '#EF6C00'


@st.cache_data
def load_land_outlines():
    """Bundled Natural Earth outlines; no map tiles, CDN, or geo service."""
    with Path(__file__).with_name("assets").joinpath("ne_110m_land.geojson").open(encoding="utf-8") as handle:
        land = json.load(handle)
    longitude, latitude = [], []
    for feature in land["features"]:
        geometry = feature["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        for polygon in polygons:
            for lon, lat in polygon[0]:
                longitude.append(lon)
                latitude.append(lat)
            longitude.append(None)
            latitude.append(None)
    return longitude, latitude


def create_dop_offline_map(observer_lat, observer_lon, dop_results=None, dop_sats_lookup=None):
    """Interactive longitude/latitude map using exclusively bundled assets."""
    fig = go.Figure()
    land_lon, land_lat = load_land_outlines()
    fig.add_trace(go.Scatter(
        x=land_lon, y=land_lat, mode="lines", fill="toself",
        line=dict(color="#78909c", width=0.6), fillcolor="#30434f",
        hoverinfo="skip", showlegend=False,
    ))
    # A fine selectable grid preserves observer picking without a tile server.
    lon_grid, lat_grid = np.meshgrid(np.arange(-180, 181, 2), np.arange(-90, 91, 2))
    fig.add_trace(go.Scatter(
        x=lon_grid.ravel(), y=lat_grid.ravel(), mode="markers",
        marker=dict(size=5, color="rgba(120,180,230,0.015)"),
        customdata=np.column_stack([lat_grid.ravel(), lon_grid.ravel()]),
        hovertemplate="Latitude %{y:.1f}°<br>Longitude %{x:.1f}°<extra>Choose observer</extra>",
        showlegend=False,
    ))
    satellites = (dop_results or {}).get("satellite_details", [])
    for sat in satellites:
        lat, lon, _ = ecef_to_geodetic(sat["ecef"])
        color = get_visibility_color(sat["elevation"])
        label = get_dop_satellite_label(sat["svn"], dop_sats_lookup or {})
        # Shortest connection across the date line, clipped by the axes.
        nearest_lon = observer_lon + circular_delta_degrees(lon, observer_lon)
        for shift in (-360, 0, 360):
            fig.add_trace(go.Scatter(
                x=[observer_lon + shift, nearest_lon + shift], y=[observer_lat, lat],
                mode="lines", line=dict(color=color, width=1), hoverinfo="skip", showlegend=False,
            ))
        fig.add_trace(go.Scatter(
            x=[lon], y=[lat], mode="markers", name=label,
            marker=dict(size=10, color=color), customdata=[[lat, lon]], showlegend=False,
            hovertemplate=f"{label}<br>Elevation {sat['elevation']:.1f}°<br>Range {sat['range']:.0f} km<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=[observer_lon], y=[observer_lat], mode="markers", name="Observer",
        marker=dict(size=13, color="#42a5f5", symbol="cross"),
        customdata=[[observer_lat, observer_lon]],
        hovertemplate="Observer<br>%{y:.4f}°, %{x:.4f}°<extra></extra>",
    ))
    fig.update_layout(
        height=520, template="plotly_dark", dragmode="pan", clickmode="event+select",
        margin=dict(l=30, r=10, t=10, b=35), plot_bgcolor="#102330",
        xaxis=dict(title="Longitude (degrees)", range=[-180, 180], dtick=30),
        yaxis=dict(title="Latitude (degrees)", range=[-90, 90], dtick=30),
        uirevision="offline-map",
    )
    return fig


def rerun_streamlit():
    """Rerun Streamlit across old and new Streamlit versions."""
    if hasattr(st, 'rerun'):
        st.rerun()
    elif hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()


def calculate_dop(parsed_data, observer_lat, observer_lon, observer_alt=0, elevation_mask=5.0, active_sats=None):
    """
    Calculate Dilution of Precision (DOP) values for any GNSS constellation
    at a specified location.
    
    Parameters
    ----------
    parsed_data : dict
        Dictionary of {svn: parsed_tle_params} for all satellites
    active_sats : dict or None
        Satellite info dict for PRN lookups (optional)
    """
    visible_sats = []
    
    # Calculate positions for all satellites
    for svn, tle_params in parsed_data.items():
        try:
            # Get satellite position
            sat_ecef = calculate_satellite_position(tle_params)
            
            # Calculate azimuth and elevation
            azimuth, elevation, range_km = calculate_azimuth_elevation(
                sat_ecef, observer_lat, observer_lon, observer_alt
            )
            
            # Check if satellite is above elevation mask
            if elevation >= elevation_mask:
                visible_sats.append({
                    'svn': svn,
                    'azimuth': azimuth,
                    'elevation': elevation,
                    'range': range_km,
                    'ecef': sat_ecef
                })
        except Exception as e:
            # Skip satellites with errors
            continue
    
    # Need at least 4 satellites for positioning
    if len(visible_sats) < 4:
        return {
            'GDOP': None,
            'PDOP': None,
            'HDOP': None,
            'VDOP': None,
            'TDOP': None,
            'visible_satellites': len(visible_sats),
            'satellite_details': visible_sats,
            'error': 'Insufficient satellites (need at least 4)'
        }
    
    # Build geometry matrix (H matrix)
    H = []
    for sat in visible_sats:
        east, north, up = ecef_to_enu(
            sat['ecef'], observer_lat, observer_lon, observer_alt
        )
        range_km = sat['range']
        
        # Unit vector pointing from observer to satellite
        H.append([east/range_km, north/range_km, up/range_km, 1.0])
    
    H = np.array(H)
    
    # Calculate DOP values from covariance matrix
    try:
        Q = np.linalg.inv(H.T @ H)
        
        # Extract DOP values
        GDOP = np.sqrt(np.trace(Q))
        PDOP = np.sqrt(Q[0,0] + Q[1,1] + Q[2,2])
        HDOP = np.sqrt(Q[0,0] + Q[1,1])
        VDOP = np.sqrt(Q[2,2])
        TDOP = np.sqrt(Q[3,3])
        
        return {
            'GDOP': GDOP,
            'PDOP': PDOP,
            'HDOP': HDOP,
            'VDOP': VDOP,
            'TDOP': TDOP,
            'visible_satellites': len(visible_sats),
            'satellite_details': visible_sats
        }
    except np.linalg.LinAlgError:
        return {
            'GDOP': None,
            'PDOP': None,
            'HDOP': None,
            'VDOP': None,
            'TDOP': None,
            'visible_satellites': len(visible_sats),
            'satellite_details': visible_sats,
            'error': 'Singular geometry matrix (satellites poorly distributed)'
        }

def main():
    st.set_page_config(page_title="GNSS Constellation Monitor", layout="wide", page_icon="🛰️")

    # This application is intentionally read-only and offline.  Every
    # constellation is loaded from one fixed, validated data snapshot before
    # any selector or analysis widget is rendered.
    global STATIC_ANALYSIS_EPOCH
    try:
        static_snapshot, STATIC_ANALYSIS_EPOCH = load_static_snapshot()
        apply_static_snapshot(static_snapshot)
        load_static_session_data(static_snapshot, STATIC_ANALYSIS_EPOCH)
    except RuntimeError as exc:
        st.error(f"Static data unavailable: {exc}")
        st.stop()
    
    # ============================
    # SIDEBAR: Constellation Selector
    # ============================
    st.sidebar.markdown("### 🌐 Constellation")
    constellation_choice = st.sidebar.selectbox(
        "Select constellation to analyze",
        options=list(CONSTELLATION_CONFIG.keys()),
        index=0,
        key="constellation_selector"
    )
    
    # Load config for selected constellation
    cfg = CONSTELLATION_CONFIG[constellation_choice]
    active_sats = cfg['sats']
    active_name = cfg['name']
    active_icon = cfg['icon']
    active_nominal_mm = cfg['nominal_mm']
    active_nominal_inc = cfg['nominal_inc']
    active_nominal_alt = cfg['nominal_alt']
    active_orbital_radius = cfg['orbital_radius']
    active_km_per_deg = cfg['km_per_deg']
    active_design_life = cfg['design_life']
    active_nominal_raan = cfg['nominal_raan']
    active_slot_angles = cfg['slot_angles']
    active_ideal_regular = cfg['ideal_regular']
    active_ideal_auxiliary = cfg['ideal_auxiliary']
    active_regular_slots = cfg.get('regular_slots', active_ideal_regular)
    active_ideal_angles = cfg.get('ideal_angles', list(range(0, 360, 45)))
    active_nominal_spacings = cfg.get('nominal_spacings', [45] * max(active_ideal_regular, 1))
    active_spacing_strategy = cfg.get('spacing_strategy', 'nominal_geometry')
    active_angular_tick = cfg.get('angular_tick', 45)
    active_gradient = cfg['gradient']
    active_accent = cfg['accent_color']
    active_plane_numbers = sorted(active_nominal_raan.keys())
    active_plane_count = len(active_plane_numbers)
    active_plane_spacing = 360 / active_plane_count if active_plane_count else 120
    
    # ============================
    # HEADER
    # ============================
    st.markdown(f"""
    <div style="padding: 20px; background: linear-gradient(90deg, {active_gradient}); border-radius: 10px; margin-bottom: 20px;">
        <h1 style="margin: 0; color: white;">🛰️ {active_name} Constellation Drift Monitor {active_icon}</h1>
        <p style="margin: 10px 0 0 0; color: #aaa; font-size: 1.1em;">Offline analysis from a fixed, common-epoch orbital-data snapshot</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ============================
    # SIDEBAR: Static snapshot
    # ============================
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗂️ Static Snapshot")
    st.sidebar.caption(f"Analysis epoch: {STATIC_ANALYSIS_EPOCH.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    st.sidebar.caption(static_snapshot.get("source", "Locally bundled orbital data"))
    
    st.sidebar.markdown("---")
    
    # ============================
    # SIDEBAR: Constellation Data
    # ============================
    st.sidebar.markdown(f"### ⚙️ {active_name} Constellation Data")
    
    # Compact status display with operational count
    total_satellite_count = len(active_sats)
    usable_count = sum(1 for sat in active_sats.values() if sat.get('status') == 'USABLE')
    not_usable_count = total_satellite_count - usable_count
    st.sidebar.metric("Operational Satellites", f"{usable_count}/{total_satellite_count}")
    
    st.sidebar.caption("🔒 Roster and operational status are frozen in this snapshot.")
    st.sidebar.caption("Service-status flags are captured repository metadata; they are not a historical status reconstruction for July 2026.")
    unavailable = static_snapshot.get("unavailable_orbital_entries", {}).get(active_name, [])
    if unavailable:
        st.sidebar.caption("No orbital record: " + ", ".join(unavailable))
    
    tle_session_key = f'tle_data_{constellation_choice}'
    fetch_time_key = f'fetch_time_{constellation_choice}'
    launch_dates_key = f'launch_dates_{constellation_choice}'

    # ============================
    # MAIN ANALYSIS
    # ============================
    if tle_session_key in st.session_state:
        tle_data = st.session_state[tle_session_key]
        fetch_time = ensure_utc_datetime(st.session_state[fetch_time_key])
        st.session_state[fetch_time_key] = fetch_time
        analysis_epoch = fetch_time
        stale_tle_svns = sorted(set(tle_data) - set(active_sats))
        if stale_tle_svns:
            tle_data = {
                svn: tle
                for svn, tle in tle_data.items()
                if svn in active_sats
            }
            st.session_state[tle_session_key] = tle_data
            stale_preview = ", ".join(stale_tle_svns[:5])
            if len(stale_tle_svns) > 5:
                stale_preview += f", +{len(stale_tle_svns) - 5} more"
            st.info(
                f"Ignoring {len(stale_tle_svns)} stale cached TLE entr"
                f"{'y' if len(stale_tle_svns) == 1 else 'ies'} not in the current {active_name} roster: "
                f"{stale_preview}. Check the bundled snapshot roster and element manifest."
            )

        st.caption(f"📡 {active_name} snapshot analysis epoch: {fetch_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        if not SKYFIELD_AVAILABLE:
            st.error(
                "Skyfield is required for this analysis. Install `skyfield` in the app's Python "
                "environment so every satellite can be propagated to the common SGP4 analysis epoch."
            )
            st.stop()
        
        # Parse all TLE data first
        raw_parsed_data = {}
        for svn, tle in tle_data.items():
            params = parse_tle_record(svn, tle)
            if params:
                raw_parsed_data[svn] = params

        if not raw_parsed_data:
            st.warning(f"No valid TLE data is available for the current {active_name} roster. Check the bundled snapshot.")
            st.stop()

        parsed_data, propagation_errors = propagate_all_tles_with_sgp4(raw_parsed_data, analysis_epoch)

        if propagation_errors:
            error_preview = ", ".join(f"{svn}: {message}" for svn, message in propagation_errors[:3])
            st.warning(
                f"SGP4 propagation failed for {len(propagation_errors)} satellite(s); "
                f"they will be excluded from analysis. {error_preview}"
            )

        if not parsed_data:
            st.error(f"No {active_name} satellites could be propagated to the common analysis epoch.")
            st.stop()

        st.caption(f"🧭 Common SGP4 analysis epoch: {analysis_epoch.strftime('%Y-%m-%d %H:%M:%S UTC')}")

        # --- Legacy (retired / standby) GPS satellites: display-only overlay ---
        # Kept in their own variables and never merged into parsed_data, so the
        # clustering, plane fit, spacing, health, and DOP paths below cannot see
        # them. A failure here degrades to an empty overlay.
        legacy_states = {}
        legacy_assignments = {}
        if active_name == 'GPS' and GPS_LEGACY_SATS:
            legacy_tle_data = st.session_state.get(
                f'legacy_tle_data_{constellation_choice}', {}
            )
            legacy_raw = {}
            for key, tle in legacy_tle_data.items():
                if key not in GPS_LEGACY_SATS:
                    continue
                params = parse_tle_record(key, tle)
                if params:
                    legacy_raw[key] = params

            legacy_states, legacy_errors = propagate_all_tles_with_sgp4(
                legacy_raw, analysis_epoch
            )
            legacy_assignments = assign_legacy_planes(legacy_states, analysis_epoch)

            missing = set(GPS_LEGACY_SATS) - set(legacy_states)
            if missing:
                names = ", ".join(
                    sorted(GPS_LEGACY_SATS[key]['name'] for key in missing)
                )
                st.caption(
                    f"🕐 Legacy overlay: {len(legacy_states)}/{len(GPS_LEGACY_SATS)} "
                    f"satellites available (no current elements for {names})"
                )

        # Cluster satellites by RAAN using k-means-like algorithm for circular data.
        # Galileo/BeiDou use 3 planes, GPS uses 6 planes.
        print("\n" + "="*80)
        print(f"RAAN-BASED PLANE CLUSTERING ANALYSIS ({active_name})")
        print("="*80)
        
        # Collect RAAN values from operational MEO satellites. Unknown-slot spares
        # still need clustering so they appear in their actual orbital planes.
        raan_data = []
        for svn, params in parsed_data.items():
            sat_info = active_sats.get(svn, {})
            slot = sat_info.get('slot', '')
            is_operational_meo = (
                sat_info.get('status') == 'USABLE'
                and sat_info.get('norad', 0) != 0
                and slot != 'Ext'
            )
            if is_operational_meo:
                raan_data.append({
                    'svn': svn,
                    'raan': params['raan'],
                    'inclination': params['inclination'],
                    'argument_latitude': params['argument_latitude']
                })
        
        print(f"Using {len(raan_data)} USABLE MEO satellites for clustering (excluding NOT_USABLE and extended/test orbits)")
        
        # Find cluster centers separated by the nominal plane spacing.
        raans = [d['raan'] for d in raan_data]
        raans_sorted = sorted(raans)
        
        # Initialize cluster centers by finding satellites roughly one nominal plane spacing apart.
        cluster_centers = []
        if len(raans_sorted) >= active_plane_count:
            center1 = raans_sorted[len(raans_sorted)//active_plane_count]
            cluster_centers = [
                min(raans, key=lambda x, target=(center1 + idx * active_plane_spacing) % 360: min(abs(x - target), 360 - abs(x - target)))
                for idx in range(active_plane_count)
            ]

        if len(cluster_centers) < active_plane_count:
            st.warning(f"Need at least {active_plane_count} usable MEO satellites to identify {active_name} orbital planes.")
            st.stop()
        
        # Iterative clustering (k-means for circular data)
        max_iterations = 10
        for iteration in range(max_iterations):
            # Assign each satellite to nearest cluster
            clusters = {idx: [] for idx in range(active_plane_count)}
            for data in raan_data:
                raan = data['raan']
                min_dist = float('inf')
                best_cluster = 0
                for idx, center in enumerate(cluster_centers):
                    dist = min(abs(raan - center), 360 - abs(raan - center))
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = idx
                clusters[best_cluster].append(data)
            
            # Update cluster centers (circular mean)
            new_centers = []
            for cluster_id in range(active_plane_count):
                if clusters[cluster_id]:
                    cluster_raans = [d['raan'] for d in clusters[cluster_id]]
                    angles_rad = np.radians(cluster_raans)
                    mean_sin = np.mean(np.sin(angles_rad))
                    mean_cos = np.mean(np.cos(angles_rad))
                    mean_raan = np.degrees(np.arctan2(mean_sin, mean_cos))
                    if mean_raan < 0:
                        mean_raan += 360
                    new_centers.append(mean_raan)
                else:
                    new_centers.append(cluster_centers[cluster_id])
            
            # Check convergence
            if all(abs(new_centers[i] - cluster_centers[i]) < 0.1 for i in range(active_plane_count)):
                break
            cluster_centers = new_centers
        
        # Anchor clusters to plane identities.
        #
        # GPS: match each cluster centre against the SPS PS Table 3.2-1 RAANs
        # propagated to the analysis epoch. Plane letters A-F are permanently
        # attached to physical orbit planes, so this is stable forever. Roster
        # votes are kept only as a cross-check -- they used to be the source of
        # truth, and because the roster had been seeded by RAAN-rank order it
        # silently mislabelled every plane by four positions.
        #
        # Other constellations: keep the roster-vote anchor.
        plane_assignment_source = 'roster'
        plane_match_diagnostics = None
        cluster_to_plane, cluster_plane_votes = assign_clusters_to_known_planes(
            clusters,
            cluster_centers,
            active_sats,
            active_plane_numbers
        )

        if active_name == 'GPS':
            nominal_plane_raans = gps_nominal_plane_raans(analysis_epoch)
            # Downstream fallbacks (drift reference, polar chart) must compare
            # against the RAAN the plane is nominally at *now*, not at the 2016
            # epoch -- that is a ~140 deg difference today.
            active_nominal_raan = nominal_plane_raans
            geometric_cluster_to_plane, plane_match_diagnostics = assign_clusters_by_nominal_raan(
                cluster_centers,
                active_plane_numbers,
                nominal_plane_raans
            )
            roster_cluster_to_plane = cluster_to_plane
            cluster_to_plane = geometric_cluster_to_plane
            plane_assignment_source = 'nominal-RAAN (SPS PS Table 3.2-1)'

            print("\nGPS plane identification vs SPS PS Table 3.2-1 (propagated):")
            for plane_num in active_plane_numbers:
                letter = GPS_PLANE_LETTERS.get(plane_num, '?')
                offset = plane_match_diagnostics['offsets'].get(plane_num)
                if offset is None:
                    print(f"  Plane {plane_num} ({letter}): unmatched")
                else:
                    print(f"  Plane {plane_num} ({letter}): nominal RAAN "
                          f"{nominal_plane_raans[plane_num]:7.2f} deg, "
                          f"cluster offset {offset:+.2f} deg")

            if plane_match_diagnostics['max_offset'] > GPS_PLANE_MATCH_WARN_DEG:
                st.warning(
                    f"⚠️ A GPS plane centre sits "
                    f"{plane_match_diagnostics['max_offset']:.1f}° from its nominal RAAN "
                    f"(warn above {GPS_PLANE_MATCH_WARN_DEG:.0f}°). Either a plane is drifting "
                    f"out of spec or the analysis epoch is wrong."
                )

            disagreements = sorted(
                plane
                for cluster_id, plane in geometric_cluster_to_plane.items()
                if roster_cluster_to_plane.get(cluster_id) not in (None, plane)
            )
            if disagreements:
                letters = ', '.join(
                    f"{GPS_PLANE_LETTERS.get(p, '?')}" for p in disagreements
                )
                st.warning(
                    f"⚠️ Roster plane labels disagree with orbital geometry for plane(s) "
                    f"{letters}. Geometry wins; `gps_status.json` needs updating "
                    f"(the frozen roster retains its captured metadata)."
                )
                print(f"ROSTER/GEOMETRY PLANE CONFLICT: planes {disagreements}")

        # Print clustering results
        print(f"\nClustering converged after {iteration + 1} iterations")
        print(f"Plane identity source: {plane_assignment_source}")
        print(f"\nIdentified {len(cluster_centers)} orbital planes:")
        for plane_num in active_plane_numbers:
            cluster_id = [k for k, v in cluster_to_plane.items() if v == plane_num][0]
            center = cluster_centers[cluster_id]
            sat_count = len(clusters[cluster_id])
            print(f"  Plane {plane_num}: Mean RAAN = {center:.2f}°, Satellites = {sat_count}")
        
        # Create detailed table
        print(f"\n{'SVN':<15} {'PRN':<8} {'RAAN (°)':<12} {'Clustered':<12} {'Original':<12} {'Incl (°)':<10} {'Arg Lat (°)':<15} {'Status':<12}")
        print("-" * 110)
        
        # Sort by clustered plane and RAAN
        cluster_assignments = {}
        for cluster_id, sat_list in clusters.items():
            plane_num = cluster_to_plane[cluster_id]
            for data in sorted(sat_list, key=lambda x: x['raan']):
                svn = data['svn']
                sat_info = active_sats[svn]
                original_plane = sat_info['plane']
                cluster_assignments[svn] = plane_num
                
                print(f"{svn:<15} {sat_info['prn']:<8} {data['raan']:<12.2f} "
                      f"Plane {plane_num:<6} Plane {original_plane:<6} "
                      f"{data['inclination']:<10.2f} {data['argument_latitude']:<15.2f} {sat_info['status']:<12}")

        # Geometry is authoritative for plane membership: publish it back onto
        # the in-memory roster so satellite typing, reference selection and slot
        # labelling all agree with the plane the satellite is physically in.
        # Keep the derived assignments in this session for inspection.
        for svn, plane_num in cluster_assignments.items():
            if svn in active_sats:
                active_sats[svn]['plane'] = plane_num
        st.session_state[f'geometry_planes_{active_name}'] = dict(cluster_assignments)

        # Summary statistics
        print("\n" + "="*80)
        print("CLUSTERING SUMMARY")
        print("="*80)
        for plane_num in active_plane_numbers:
            cluster_id = [k for k, v in cluster_to_plane.items() if v == plane_num][0]
            sats_in_plane = clusters[cluster_id]
            usable = sum(1 for d in sats_in_plane if active_sats[d['svn']]['status'] == 'USABLE')
            center = cluster_centers[cluster_id]
            raan_range = [d['raan'] for d in sats_in_plane]
            if raan_range:
                raan_std = np.std(raan_range)
                print(f"Plane {plane_num}:")
                print(f"  Mean RAAN: {center:.2f}°")
                print(f"  RAAN Std Dev: {raan_std:.2f}°")
                print(f"  Total Satellites: {len(sats_in_plane)}")
                print(f"  Operational (USABLE): {usable}")
                print(f"  RAAN Range: {min(raan_range):.2f}° - {max(raan_range):.2f}°")
                print()
        
        # Check plane separation
        print("Plane Separation Analysis:")
        centers_sorted = sorted(cluster_centers)
        for i in range(active_plane_count):
            sep = (centers_sorted[(i+1)%active_plane_count] - centers_sorted[i]) % 360
            next_plane_num = active_plane_numbers[(i+1)%active_plane_count]
            print(f"  Plane {active_plane_numbers[i]} to Plane {next_plane_num}: {sep:.2f} deg")
            continue
            print(f"  Plane {i+1} to Plane {(i+1)%3 + 1}: {sep:.2f}°")
        print("="*80 + "\n")
        
        # Use clustered assignments for further analysis
        plane_mean_raans = {plane_num: cluster_centers[[k for k, v in cluster_to_plane.items() if v == plane_num][0]] 
                           for plane_num in active_plane_numbers}
        
        # Second pass: Calculate drift relative to plane mean RAAN (using clustered assignments)
        results = []
        for svn, params in parsed_data.items():
            sat_info = active_sats[svn]
            # Use clustered plane assignment instead of pre-defined
            plane_num = cluster_assignments.get(svn, sat_info['plane'])
            tle_epoch = params['tle_epoch']
            epoch_delta_days = (tle_epoch - analysis_epoch).total_seconds() / 86400
            
            drift = calculate_drift(
                params, plane_num, plane_mean_raans.get(plane_num),
                nominal_mean_motion=active_nominal_mm,
                nominal_inclination=active_nominal_inc,
                orbital_radius=active_orbital_radius,
                km_per_degree=active_km_per_deg,
                nominal_raan_map=active_nominal_raan
            )
            results.append({
                'SVN': svn,
                'PRN': sat_info['prn'],
                'Slot': sat_info['slot'],
                'Plane': f"Plane {plane_num}" if plane_num > 0 else "Non-nominal",
                'Status': sat_info.get('status', 'USABLE'),
                'Type': get_satellite_type(sat_info['slot'], plane_num, active_regular_slots),
                'TLE Epoch (UTC)': tle_epoch.strftime('%Y-%m-%d %H:%M:%S'),
                'Δ Common Epoch (days)': round(epoch_delta_days, 2),
                '|Δ Common Epoch| (days)': round(abs(epoch_delta_days), 2),
                'Mean Motion (rev/day)': params['mean_motion'],
                'Δn (rev/day)': drift['delta_mean_motion'],
                'In-Plane Drift (km/day)': drift['in_plane_drift_km_day'],
                'Period (hours)': drift['orbital_period_hours'],
                'RAAN (°)': params['raan'],
                'ΔRAAN (°)': drift['delta_raan'],
                'Inclination (°)': params['inclination'],
                'ΔInclination (°)': drift['delta_inclination'],
                'Cross-Track (km)': drift['cross_track_km'],
                'TLE Mean Anomaly (°)': drift['mean_anomaly'],
                'Argument of Latitude (°)': params['argument_latitude'],
                'In-Plane OK': '✅' if drift['in_plane_ok'] else '❌',
                'Across-Track OK': '✅' if drift['across_track_ok'] else '❌'
            })
        
        df = pd.DataFrame(results)
        
        # Filter to only operational (USABLE) satellites for analysis
        df = df[df['Status'] == 'USABLE']
        
        # Summary metrics
        st.markdown("---")
        st.markdown(f"### 📡 {active_name} Quick Status Overview")
        
        analyzed_count = len(df)
        tle_coverage_pct = (analyzed_count / usable_count * 100) if usable_count > 0 else 0

        # A seventh column appears only when there is a legacy overlay to put in
        # it. Widening the row unconditionally would shrink the existing six
        # metrics on the three constellations that have no legacy roster.
        show_legacy_metric = bool(active_name == 'GPS' and legacy_assignments)
        if show_legacy_metric:
            col1, col2, col3, col4, col5, col6, legacy_metric_column = st.columns(7)
        else:
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            legacy_metric_column = None

        with col1:
            st.metric("🛰️ Total Satellites", total_satellite_count)
        with col2:
            not_usable_delta = f"-{not_usable_count} not usable" if not_usable_count else None
            st.metric("🟢 Operational (DLR)", f"{usable_count}/{total_satellite_count}", not_usable_delta)
        with col3:
            st.metric("📡 TLEs Analyzed", f"{analyzed_count}/{usable_count}", f"{tle_coverage_pct:.0f}%")
        with col4:
            in_plane_ok = df['In-Plane OK'].value_counts().get('✅', 0)
            pct = (in_plane_ok/analyzed_count*100) if analyzed_count > 0 else 0
            st.metric("📈 In-Plane OK", f"{in_plane_ok}/{analyzed_count}", f"{pct:.0f}%")
        with col5:
            across_ok = df['Across-Track OK'].value_counts().get('✅', 0)
            pct = (across_ok/analyzed_count*100) if analyzed_count > 0 else 0
            st.metric("🌍 Across-Track OK", f"{across_ok}/{analyzed_count}", f"{pct:.0f}%")
        with col6:
            both_ok = len(df[(df['In-Plane OK'] == '✅') & (df['Across-Track OK'] == '✅')])
            pct = (both_ok/analyzed_count*100) if analyzed_count > 0 else 0
            st.metric("✅ Both OK", f"{both_ok}/{analyzed_count}", f"{pct:.0f}%")
        if legacy_metric_column is not None:
            resident = [a for a in legacy_assignments.values()
                        if a.get('plane') is not None]
            holding = sum(1 for a in resident
                          if a['tier'] == GPS_LEGACY_TIER_HOLDING)
            drifting = len(resident) - holding
            with legacy_metric_column:
                st.metric("🕐 Legacy in-plane", f"{len(resident)}",
                          f"{holding} holding · {drifting} drifting",
                          delta_color="off")

        st.markdown("---")

        # Tabs for different views
        tab_list = ["🛰️ Constellation Map", "📊 Overview", "📈 In-Plane Drift", "🌍 Across-Track Drift", "📡 DOP Calculator"]
        
        tabs = st.tabs(tab_list)
        tab_idx = 0
        
        # Store plane health data for use across tabs
        plane_health_data = {}
        plane_spacings_for_symmetry = {}
        
        with tabs[tab_idx]:  # Constellation Map (now first)
            tab_idx += 1
            
            # Placeholder for health summary - will be populated after plane analysis
            health_summary_placeholder = st.container()
            
                # Create constellation plot with common-epoch SGP4 geometry (only operational satellites)
            operational_df = df  # All satellites in df are operational
            if not operational_df.empty:
                st.markdown("#### 3D Globe View - Orbital Planes and Satellites")
                fig_globe = create_3d_constellation_globe(
                    parsed_data,
                    operational_df,
                    plane_mean_raans,
                    active_sats,
                    active_orbital_radius,
                    active_nominal_inc,
                    active_name,
                    active_plane_numbers,
                    legacy_states=legacy_states if active_name == 'GPS' else None,
                    legacy_assignments=legacy_assignments if active_name == 'GPS' else None,
                )
                st.plotly_chart(fig_globe, use_container_width=True)
                st.markdown("")

                # Create individual plots for each clustered plane
                for plane_num in active_plane_numbers:
                    plane_name = f"Plane {plane_num}"
                    plane_df_initial = operational_df[operational_df['Plane'] == f"Plane {plane_num}"].copy()
                    
                    if plane_df_initial.empty:
                        continue
                    
                    # Create visual separator and container for each plane
                    st.markdown("---")
                    st.markdown(f"## 🛰️ {plane_name}")
                    
                    # Select a slotted reference satellite. Spares can be plotted,
                    # but they should not define the nominal-slot scoring frame.
                    ref_svn = choose_plane_reference_svn(
                        plane_df_initial,
                        active_sats,
                        active_regular_slots,
                        active_slot_angles,
                        excluded_slot=(
                            GPS_EXPANDABLE_BASELINE_SLOT.get(plane_num)
                            if active_name == 'GPS' else None
                        )
                    )
                    if ref_svn is None:
                        continue
                    
                    # Filter satellites in this plane by epoch (adaptive filtering with fallback)
                    plane_svns = plane_df_initial['SVN'].tolist()
                    plane_parsed_data = {svn: parsed_data[svn] for svn in plane_svns if svn in parsed_data}
                    
                    min_sats = len(plane_svns)
                    filtered_plane_data, epoch_metadata = filter_satellites_by_epoch(
                        plane_parsed_data, ref_svn, max_days_diff=3, min_satellites=min_sats
                    )
                    
                    # Filter dataframe to only include satellites within epoch tolerance
                    filtered_svns = list(filtered_plane_data.keys())
                    plane_df = plane_df_initial[plane_df_initial['SVN'].isin(filtered_svns)].copy()
                    
                    if plane_df.empty:
                        st.warning(f"⚠️ No satellites found for {plane_name}")
                        continue
                    
                    # Get reference satellite info
                    ref_epoch = filtered_plane_data[ref_svn]['epoch']
                    excluded_count = epoch_metadata['excluded_count']
                    expanded_window = epoch_metadata['expanded_window']
                    epoch_warnings = epoch_metadata['epoch_warnings']
                    
                    # Show epoch quality warning if window was expanded
                    if expanded_window and expanded_window != False:
                        if expanded_window == 999:
                            st.warning(f"⚠️ Using all available TLE data (some satellites have epochs >30 days apart)")
                        else:
                            st.info(f"ℹ️ Epoch window expanded to ±{expanded_window} days to include sufficient satellites")
                    elif excluded_count > 0:
                        st.caption(f"ℹ️ {excluded_count} satellite(s) excluded (TLE epochs >3 days from reference)")
                    
                    # Debug info
                    all_plane_svns = set(plane_df_initial['SVN'].tolist())
                    included_svns = set(filtered_svns)
                    excluded_svns = all_plane_svns - included_svns
                    
                    if excluded_svns:
                        with st.expander(f"🔍 Debug: Epoch filtering details ({len(included_svns)} included, {len(excluded_svns)} excluded)"):
                            st.caption(f"**Reference satellite:** {ref_svn} (epoch: {ref_epoch.strftime('%Y-%m-%d %H:%M:%S')})")
                            st.caption(f"**Included satellites:** {', '.join(sorted(included_svns))}")
                            st.caption(f"**Excluded satellites:** {', '.join(sorted(excluded_svns))}")
                            
                            if excluded_svns and plane_parsed_data:
                                st.caption("**Epoch differences for excluded satellites:**")
                                for svn in sorted(excluded_svns):
                                    if svn in plane_parsed_data:
                                        sat_epoch = plane_parsed_data[svn]['epoch']
                                        diff = abs((sat_epoch - ref_epoch).total_seconds() / 86400)
                                        st.caption(f"  - {svn}: {diff:.1f} days difference (epoch: {sat_epoch.strftime('%Y-%m-%d %H:%M:%S')})")
                    
                    # Use common-epoch SGP4 argument of latitude for in-plane geometry.
                    ref_argument_latitude = filtered_plane_data[ref_svn]['argument_latitude']
                    rel_longitudes = {
                        svn: plane_relative_argument_latitude(
                            filtered_plane_data[svn]['argument_latitude'],
                            ref_argument_latitude
                        )
                        for svn in filtered_svns
                        if svn in filtered_plane_data and 'argument_latitude' in filtered_plane_data[svn]
                    }
                    
                    plane_mean_raan = plane_mean_raans.get(plane_num, active_nominal_raan.get(plane_num, 0))
                    
                    # Sort by relative argument of latitude
                    plane_df = plane_df.copy()
                    plane_df['rel_arg_lat_calc'] = plane_df['SVN'].map(rel_longitudes)
                    plane_df = plane_df.sort_values('rel_arg_lat_calc').reset_index(drop=True)
                    
                    if len(plane_df) > 0:
                        ref_rows = plane_df[plane_df['SVN'] == ref_svn]
                        ref_sat = ref_rows.iloc[0] if not ref_rows.empty else plane_df.iloc[0]
                        ref_svn = ref_sat['SVN']
                        ref_argument_latitude = filtered_plane_data[ref_svn]['argument_latitude']
                        ref_slot = active_sats.get(ref_svn, {}).get('slot', '')
                        ref_nominal_angle = get_nominal_angle(ref_slot, active_slot_angles)
                        ref_type = get_satellite_type(ref_slot, plane_num, active_regular_slots)
                        plot_angle_offset = (
                            ref_nominal_angle
                            if ref_type == 'Regular' and ref_nominal_angle is not None
                            else 180
                        )

                        # GPS: fit the in-plane frame to the plane's nominal
                        # geometry instead of pinning it to one satellite's slot
                        # label. Anchoring on a reference satellite forced that
                        # satellite's deviation to exactly 0.00, hid any
                        # common-mode in-plane error, and made every score depend
                        # on which satellite happened to be chosen. It also
                        # silently depended on the roster slot label being right.
                        if active_name == 'GPS':
                            expected_positions = GPS_EXPECTED_POSITIONS_BY_PLANE.get(plane_num) or {}
                            fit_angles = [
                                rel_longitudes[svn] for svn in plane_df['SVN']
                                if svn in rel_longitudes
                                and active_sats.get(svn, {}).get('status') == 'USABLE'
                            ]
                            if expected_positions and fit_angles:
                                plot_angle_offset = fit_plane_phase_to_nominal(
                                    fit_angles, list(expected_positions.values())
                                )

                        # Calculate relative values for each satellite
                        plot_data = []
                        for _, row in plane_df.iterrows():
                            svn = row['SVN']
                            sat_info = active_sats[svn]
                            
                            # Calculate relative RAAN
                            relative_raan = row['RAAN (°)'] - plane_mean_raan
                            if relative_raan > 180:
                                relative_raan -= 360
                            elif relative_raan < -180:
                                relative_raan += 360
                            
                            # Use common-epoch propagated relative argument of latitude
                            relative_argument_latitude = rel_longitudes.get(svn, 0)
                            plot_angle = plane_plot_angle(relative_argument_latitude, plot_angle_offset)
                            
                            # Determine satellite type
                            sat_type = get_satellite_type(sat_info['slot'], plane_num, active_regular_slots)
                            
                            # Get epoch info for this satellite
                            sat_epoch = filtered_plane_data[svn]['epoch']
                            epoch_diff_days = (sat_epoch - ref_epoch).total_seconds() / 86400
                            
                            plot_data.append({
                                'SVN': svn,
                                'Slot': sat_info['slot'],
                                'PRN': sat_info['prn'],
                                'Relative_RAAN': relative_raan,
                                'Relative_Argument_Latitude': relative_argument_latitude,
                                'Plot_Angle': plot_angle,
                                'Type': sat_type,
                                'Status': sat_info.get('status', 'USABLE'),
                                'Label': sat_info['slot'] if sat_info['slot'] != 'Unk' else sat_info['prn'],
                                'TLE_Mean_Anomaly': row['TLE Mean Anomaly (°)'],
                                'Argument_Latitude': row['Argument of Latitude (°)'],
                                'RAAN': row['RAAN (°)'],
                                'Epoch': sat_epoch,
                                'Epoch_Diff_Days': epoch_diff_days
                            })
                        
                        plot_df = pd.DataFrame(plot_data)
                        plot_df = assign_polar_marker_radii(plot_df)

                        # GPS: dynamic geometric classification (occupancy map).
                        # Overrides slot-name-based Type ('Regular'/'Auxiliary'/'Spare')
                        # with a classification derived from where each satellite
                        # physically sits vs the 6 expected positions per plane
                        # (4 baseline + 2 expansion). See build_gps_plane_occupancy.
                        gps_occupancy = None
                        if active_name == 'GPS':
                            # Launch dates settle a contested position: the
                            # incumbent keeps the slot, the newer satellite
                            # parked beside it is the replacement in waiting.
                            gps_occupancy = build_gps_plane_occupancy(
                                plot_df, plane_num,
                                launch_dates=st.session_state.get(launch_dates_key, {})
                            )
                            plot_df['GPS_Assigned_Label'] = plot_df['Slot']
                            plot_df['GPS_Expected_Angle'] = np.nan
                            plot_df['GPS_Deviation'] = np.nan
                            plot_df['GPS_Classification'] = 'Unclassified'
                            for record in gps_occupancy['baseline']:
                                mask = plot_df['SVN'] == record['svn']
                                plot_df.loc[mask, 'Type'] = 'Regular'
                                plot_df.loc[mask, 'GPS_Assigned_Label'] = record['assigned_label']
                                plot_df.loc[mask, 'GPS_Expected_Angle'] = record['expected_angle']
                                plot_df.loc[mask, 'GPS_Deviation'] = record['deviation']
                                plot_df.loc[mask, 'GPS_Classification'] = 'Baseline'
                            for record in gps_occupancy['expansion']:
                                mask = plot_df['SVN'] == record['svn']
                                plot_df.loc[mask, 'Type'] = 'Auxiliary'
                                plot_df.loc[mask, 'GPS_Assigned_Label'] = record['assigned_label']
                                plot_df.loc[mask, 'GPS_Expected_Angle'] = record['expected_angle']
                                plot_df.loc[mask, 'GPS_Deviation'] = record['deviation']
                                plot_df.loc[mask, 'GPS_Classification'] = (
                                    'Expansion-Aft' if record.get('position') == 'aft' else 'Expansion-Fore'
                                )
                            for record in gps_occupancy['unclassified']:
                                mask = plot_df['SVN'] == record['svn']
                                plot_df.loc[mask, 'Type'] = 'Spare'
                                plot_df.loc[mask, 'GPS_Assigned_Label'] = 'Auxiliary'
                                plot_df.loc[mask, 'GPS_Expected_Angle'] = record['nearest_expected_angle']
                                plot_df.loc[mask, 'GPS_Deviation'] = record['nearest_deviation']
                                plot_df.loc[mask, 'GPS_Classification'] = 'Auxiliary'

                            # Collapse onto the four SPS PS slots and work out
                            # which position each auxiliary is shadowing.
                            gps_slot_summary = summarize_gps_plane_slots(gps_occupancy)
                            shadowed_by_svn = {
                                aux['svn']: aux for aux in gps_slot_summary['auxiliaries']
                            }
                            plot_df['GPS_Shadows'] = plot_df['SVN'].map(
                                lambda svn: (shadowed_by_svn.get(svn) or {}).get('shadows')
                            )
                            plot_df['GPS_Shadow_Offset'] = plot_df['SVN'].map(
                                lambda svn: (shadowed_by_svn.get(svn) or {}).get('offset_deg')
                            )
                            # Chart labels must show where a satellite actually
                            # is, not the slot name its roster entry carries --
                            # a marker reading "A04" while coloured as an extra
                            # is the single most confusing thing on this plot.
                            plot_df['Marker_Label'] = [
                                row['PRN'] if row['GPS_Classification'] == 'Auxiliary'
                                else row['GPS_Assigned_Label']
                                for _, row in plot_df.iterrows()
                            ]
                        else:
                            gps_slot_summary = None
                            plot_df['Marker_Label'] = plot_df['Label']

                        # Galileo/BeiDou/GLONASS: an Auxiliary/Spare satellite can be
                        # maneuvered into a baseline slot's geometric position when the
                        # slot's usual occupant goes NOT_USABLE (e.g. GSAT0232/A17
                        # covering Galileo plane 1's vacant A08). Detect this from the
                        # live TLE geometry so the marker tooltip and caption say which
                        # slot it's covering, without changing Type/score bucket.
                        plot_df = apply_baseline_fill_labels(
                            plot_df, active_slot_angles, active_regular_slots, active_name
                        )

                        print_argument_of_latitude_debug(
                            active_name,
                            plane_num,
                            plot_df,
                            filtered_plane_data,
                            ref_svn,
                            analysis_epoch
                        )
                        
                        # Define colors and per-constellation display vocabulary.
                        # GPS has no auxiliary tier: extras are either at spec
                        # expansion positions or are active replacements/spares.
                        color_map = {
                            ('Regular', 'USABLE'): '#2E7D32',      # Green
                            ('Auxiliary', 'USABLE'): '#FFD700',    # Gold
                            ('Spare', 'USABLE'): '#29B6F6',        # Blue
                            ('Regular', 'NOT_USABLE'): '#E53935',  # Red
                            ('Auxiliary', 'NOT_USABLE'): '#E53935', # Red
                            ('Spare', 'NOT_USABLE'): '#E53935'     # Red
                        }
                        if active_name == 'GPS':
                            color_map[('Auxiliary', 'USABLE')] = '#FF8C00'  # Orange, pairs with expansion markers
                            # GPS has exactly three categories. 3.2 defines
                            # baseline slots and expandable slots; anything else
                            # is an auxiliary, which the spec says has no slot at
                            # all. "Spare" was never a GPS concept.
                            type_display = {
                                'Regular': 'Baseline slot',
                                'Auxiliary': 'Expansion position',
                                'Spare': 'Auxiliary (no defined slot)',
                            }
                            symbol_map = {'Regular': 'circle', 'Auxiliary': 'triangle-up', 'Spare': 'square'}
                        else:
                            type_display = {'Regular': 'Regular', 'Auxiliary': 'Auxiliary', 'Spare': 'Spare'}
                            symbol_map = {'Regular': 'circle', 'Auxiliary': 'diamond', 'Spare': 'square'}
                        
                        # Create circular/polar plot
                        st.markdown("#### Circular View - Satellite Positions in Orbital Plane")
                        
                        fig_polar = go.Figure()
                        
                        for (sat_type, status), color in color_map.items():
                            mask = (plot_df['Type'] == sat_type) & (plot_df['Status'] == status)
                            subset = plot_df[mask]
                            
                            if len(subset) > 0:
                                marker_symbol = symbol_map.get(sat_type, 'circle')
                                legend_name = f"{type_display.get(sat_type, sat_type)} {'Operational' if status == 'USABLE' else 'Not-in-Service'}"
                                
                                theta_values = subset['Plot_Angle']
                                if active_name == 'GPS':
                                    detail_hover = []
                                    for _, row in subset.iterrows():
                                        parts = [f"{row['PRN']} ({row['SVN']})"]
                                        if row['GPS_Classification'] == 'Auxiliary':
                                            if pd.notna(row.get('GPS_Shadows')):
                                                parts.append(
                                                    f"Auxiliary shadowing {row['GPS_Shadows']} "
                                                    f"({row['GPS_Shadow_Offset']:+.2f}°) — likely "
                                                    f"replacement in waiting"
                                                )
                                            else:
                                                parts.append(
                                                    "Auxiliary — no defined slot (SPS PS 3.2)"
                                                )
                                        else:
                                            parts.append(f"Roster slot {row['Slot']}")
                                            if pd.notna(row.get('GPS_Deviation')):
                                                parts.append(
                                                    f"Deviation from nominal: "
                                                    f"{row['GPS_Deviation']:+.2f}°"
                                                )
                                        detail_hover.append("<br>".join(parts))
                                else:
                                    detail_hover = subset['Baseline_Fill_Label'].apply(
                                        lambda v: f"Filling baseline slot {v}" if pd.notna(v) else ''
                                    ).tolist()

                                fig_polar.add_trace(go.Scatterpolar(
                                    r=subset['Marker_Radius'],
                                    theta=theta_values,
                                    mode='markers+text',
                                    marker=dict(
                                        size=18,
                                        color=color,
                                        symbol=marker_symbol,
                                        line=dict(width=2, color='white')
                                    ),
                                    text=subset['Marker_Label'],
                                    customdata=detail_hover,
                                    textposition='middle center',
                                    textfont=dict(size=9, color='white', family='Arial Black'),
                                    name=legend_name,
                                    hovertemplate="<b>%{text}</b><br>" +
                                                 "Angle: %{theta:.1f}°<br>" +
                                                 "%{customdata}" +
                                                 "<extra></extra>"
                                ))
                        
                        # Add reference marker
                        ref_row = plot_df[plot_df['SVN'] == ref_svn].iloc[0]
                        ref_theta = ref_row['Plot_Angle']
                        fig_polar.add_trace(go.Scatterpolar(
                            r=[1.15],
                            theta=[ref_theta],
                            mode='markers+text',
                            marker=dict(
                                size=22,
                                color='gold',
                                symbol='star',
                                line=dict(width=2, color='orange')
                            ),
                            text=['REF'],
                            textfont=dict(size=8, color='black', family='Arial Black'),
                            name='Reference Satellite',
                            hovertemplate=f"<b>Reference: {ref_svn}</b><br>" +
                                         f"Slot: {ref_row['Slot']}<br>" +
                                         "<extra></extra>"
                        ))

                        # Legacy (retired / standby) satellites, drawn on an
                        # inner band. The radial axis runs [0, 1.22]; active
                        # markers occupy 0.79-1.21 and nominal positions 0.95,
                        # so 0.62 cannot be mistaken for either. Deliberately
                        # subordinate styling: these are not part of the
                        # constellation being scored.
                        legacy_plane_df = pd.DataFrame()
                        if active_name == 'GPS' and legacy_assignments:
                            legacy_plane_df = build_legacy_plane_frame(
                                legacy_states, legacy_assignments, plane_num,
                                ref_argument_latitude, plot_angle_offset
                            )
                        if not legacy_plane_df.empty:
                            legacy_hover = []
                            for _, legacy_row in legacy_plane_df.iterrows():
                                slot_note = (
                                    f"Nearest slot: {legacy_row['Nearest_Slot']} "
                                    f"({legacy_row['Slot_Offset']:+.1f}°)"
                                    if legacy_row['Nearest_Slot'] else 'Nearest slot: n/a'
                                )
                                tier_note = (
                                    f"⚠ drifting ({legacy_row['RAAN_Offset']:+.1f}° "
                                    f"from nominal RAAN)"
                                    if legacy_row['Tier'] == GPS_LEGACY_TIER_DRIFTING
                                    else f"holding ({legacy_row['RAAN_Offset']:+.1f}° "
                                         f"from nominal RAAN)"
                                )
                                legacy_hover.append(
                                    f"NORAD {legacy_row['NORAD']}<br>"
                                    f"SATCAT: {legacy_row['SATCAT_Status']}<br>"
                                    f"Launched {legacy_row['Launch_Date']}<br>"
                                    f"{tier_note}<br>{slot_note}"
                                )

                            fig_polar.add_trace(go.Scatterpolar(
                                r=[0.62] * len(legacy_plane_df),
                                theta=legacy_plane_df['Plot_Angle'],
                                mode='markers',
                                marker=dict(
                                    size=12,
                                    color='rgba(158, 158, 158, 0.85)',
                                    symbol='diamond-open',
                                    line=dict(width=2, color='#616161')
                                ),
                                text=legacy_plane_df['Name'],
                                customdata=legacy_hover,
                                name='Legacy (retired / standby)',
                                hovertemplate="<b>%{text}</b><br>"
                                              "Angle: %{theta:.1f}°<br>"
                                              "%{customdata}"
                                              "<extra></extra>"
                            ))

                        # Add constellation-specific ideal baseline geometry positions.
                        if active_spacing_strategy == 'gps_nominal':
                            plane_ideal = cfg.get('ideal_angles_by_plane', {}).get(plane_num)
                            ideal_angles = np.array(plane_ideal, dtype=float) if plane_ideal else np.array(active_ideal_angles, dtype=float)
                            plane_nom_spacings = cfg.get('nominal_spacings_by_plane', {}).get(plane_num, active_nominal_spacings)
                            ideal_spacing_label = " / ".join(f"{s:g} deg" for s in plane_nom_spacings)
                        else:
                            ideal_angles = np.array(active_ideal_angles, dtype=float)
                            ideal_spacing_label = " / ".join(f"{angle:g} deg" for angle in active_nominal_spacings)
                        fig_polar.add_trace(go.Scatterpolar(
                            r=[0.95]*len(ideal_angles),
                            theta=ideal_angles,
                            mode='markers',
                            marker=dict(
                                size=8,
                                color='rgba(173, 216, 230, 0.5)',
                                symbol='circle-open',
                                line=dict(width=2, color='lightblue')
                            ),
                            name='Ideal 45° Spacing',
                            hovertemplate='Ideal Position: %{theta:.0f}°<extra></extra>'
                        ))
                        
                        fig_polar.data[-1].name = 'Nominal Geometry Positions'
                        fig_polar.data[-1].hovertemplate = 'Nominal Position: %{theta:.0f} deg<extra></extra>'

                        # GPS: an expanded slot's centre is *meant* to be empty --
                        # its pair sits fore and aft of it. Drawn as a plain open
                        # circle it reads as a vacancy, which is the opposite of
                        # the truth. Style each nominal centre by its slot state.
                        if active_name == 'GPS' and gps_slot_summary:
                            slot_labels = GPS_BASELINE_SLOT_ORDER_BY_PLANE.get(plane_num) or []
                            states = gps_slot_summary['slot_states']
                            state_style = {
                                'baseline': ('lightblue', 'occupied'),
                                'transitional': ('#FF8C00', 'occupied, expansion in progress'),
                                'expanded': ('#7E57C2', 'expanded — centre intentionally vacant'),
                                'partial': ('#E53935', 'PARTIAL — lost one of the expanded pair'),
                                'vacant': ('#E53935', 'VACANT'),
                            }
                            if len(slot_labels) == len(ideal_angles):
                                colours, notes = [], []
                                for label in slot_labels:
                                    colour, note = state_style.get(
                                        states.get(label, 'vacant'), ('lightblue', '')
                                    )
                                    colours.append(colour)
                                    notes.append(f"{label}: {note}")
                                fig_polar.data[-1].marker.line.color = colours
                                fig_polar.data[-1].customdata = notes
                                fig_polar.data[-1].hovertemplate = (
                                    '%{customdata}<br>Nominal centre: %{theta:.0f} deg<extra></extra>'
                                )
                                fig_polar.data[-1].name = 'Nominal Slot Centres'

                        # GPS-only: overlay the 2 expansion positions (aft/fore) for
                        # this plane's expandable baseline slot as open triangles.
                        if active_name == 'GPS':
                            exp_positions = GPS_EXPECTED_POSITIONS_BY_PLANE.get(plane_num, {})
                            expandable = GPS_EXPANDABLE_BASELINE_SLOT.get(plane_num)
                            if expandable:
                                aft_label, fore_label = expandable + 'A', expandable + 'F'
                                exp_labels = [lbl for lbl in (aft_label, fore_label) if lbl in exp_positions]
                                if exp_labels:
                                    fig_polar.add_trace(go.Scatterpolar(
                                        r=[0.95] * len(exp_labels),
                                        theta=[exp_positions[lbl] for lbl in exp_labels],
                                        mode='markers',
                                        marker=dict(
                                            size=12,
                                            color='rgba(255, 200, 100, 0.7)',
                                            symbol='triangle-up-open',
                                            line=dict(width=2, color='#FF8C00'),
                                        ),
                                        name='Expansion Positions',
                                        customdata=exp_labels,
                                        hovertemplate='Expansion %{customdata}: %{theta:.0f} deg<extra></extra>',
                                    ))

                        fig_polar.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=False,
                                    range=[0, 1.2]
                                ),
                                angularaxis=dict(
                                    direction='clockwise',
                                    rotation=90,
                                    tickmode='linear',
                                    tick0=0,
                                    dtick=active_angular_tick,
                                    gridcolor='lightgray'
                                ),
                                bgcolor='white'
                            ),
                            showlegend=True,
                            title=f"Circular View - Satellite Distribution<br>" +
                                  f"<sub>Reference: {ref_svn} (gold star) | Light blue circles = Ideal 45° positions</sub>",
                            height=550,
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=1.01
                            )
                        )
                        
                        fig_polar.update_layout(
                            title=f"Circular View - Satellite Distribution<br>"
                                  f"<sub>Reference: {ref_svn} (gold star) | Light blue circles = nominal geometry positions ({ideal_spacing_label})</sub>"
                        )

                        polar_subtitle = (
                            f"Reference: {ref_svn} (gold star) | "
                            f"Light blue circles = nominal geometry positions ({ideal_spacing_label})"
                        )
                        fig_polar.update_layout(
                            polar=dict(
                                domain=dict(x=[0.04, 0.78], y=[0.0, 0.92]),
                                radialaxis=dict(
                                    visible=False,
                                    range=[0, 1.22]
                                ),
                                angularaxis=dict(
                                    direction='clockwise',
                                    rotation=90,
                                    tickmode='linear',
                                    tick0=0,
                                    dtick=active_angular_tick,
                                    gridcolor='lightgray',
                                    tickfont=dict(size=11)
                                ),
                                bgcolor='white'
                            ),
                            showlegend=True,
                            title=dict(
                                text=f"Circular View - Satellite Distribution<br><sub>{polar_subtitle}</sub>",
                                x=0.5,
                                y=0.98,
                                xanchor='center',
                                yanchor='top',
                                font=dict(size=15)
                            ),
                            height=620,
                            margin=dict(l=20, r=230, t=95, b=35),
                            legend=dict(
                                yanchor="top",
                                y=0.94,
                                xanchor="left",
                                x=0.82,
                                bgcolor='rgba(0,0,0,0)',
                                font=dict(size=11)
                            )
                        )

                        with st.container():
                            st.plotly_chart(fig_polar, use_container_width=True)

                        if not legacy_plane_df.empty:
                            plane_letter = GPS_PLANE_LETTERS.get(plane_num, str(plane_num))
                            with st.expander(
                                f"🕐 Legacy satellites in plane {plane_letter} "
                                f"({len(legacy_plane_df)})",
                                expanded=False
                            ):
                                st.caption(
                                    "Retired or standby hardware still resident in this "
                                    "plane. Excluded from every metric on this page — "
                                    "spacing, slot occupancy, plane health, and DOP all "
                                    "ignore these satellites."
                                )
                                display = pd.DataFrame({
                                    'Name': legacy_plane_df['Name'],
                                    'NORAD': legacy_plane_df['NORAD'],
                                    'PRN (historic)': legacy_plane_df['PRN'].replace('', '—'),
                                    'SATCAT status': legacy_plane_df['SATCAT_Status'],
                                    'Launched': legacy_plane_df['Launch_Date'],
                                    'Tier': legacy_plane_df['Tier'],
                                    'RAAN offset (°)': legacy_plane_df['RAAN_Offset'].round(2),
                                    'Arg. latitude (°)': legacy_plane_df['Argument_Latitude'].round(2),
                                    'Nearest slot': legacy_plane_df['Nearest_Slot'],
                                    'Offset from slot (°)': legacy_plane_df['Slot_Offset'].round(2),
                                })
                                st.dataframe(display, use_container_width=True,
                                             hide_index=True)

                        spacing_expander_title = (
                            "Baseline Coverage Analysis (Regular Satellites)"
                            if active_spacing_strategy == 'coverage'
                            else "GPS Nominal Geometry Analysis (Regular Satellites)"
                            if active_spacing_strategy == 'gps_nominal'
                            else "Nominal Geometry Analysis (Regular Satellites)"
                        )
                        with st.expander(spacing_expander_title, expanded=False):
                            sat_angles = []
                            for _, row_data in plot_df.iterrows():
                                if row_data['Type'] == 'Regular':
                                    theta = row_data['Plot_Angle']
                                    sat_angles.append({
                                        'SVN': row_data['SVN'],
                                        'Slot': row_data['Slot'],
                                        'PRN': row_data['PRN'],
                                        'Angle': theta,
                                        'Type': row_data['Type']
                                    })
                            
                            sat_angles_sorted = sorted(sat_angles, key=lambda x: x['Angle'])
                            
                            spacing_data = []
                            gps_pattern_result = None
                            if active_spacing_strategy == 'coverage':
                                coverage_health = calculate_coverage_spacing_health(
                                    len(sat_angles_sorted),
                                    active_ideal_regular
                                )
                                if coverage_health >= 95:
                                    health_status = '🟢 Excellent'
                                elif coverage_health >= 85:
                                    health_status = '🟡 Good'
                                elif coverage_health >= 70:
                                    health_status = '🟠 Fair'
                                else:
                                    health_status = '🔴 Poor'

                                for i, sat in enumerate(sat_angles_sorted):
                                    next_sat = sat_angles_sorted[(i + 1) % len(sat_angles_sorted)]
                                    actual_spacing = (next_sat['Angle'] - sat['Angle']) % 360
                                    spacing_data.append({
                                        'Slot': sat['Slot'],
                                        'SVN': sat['SVN'],
                                        'PRN': sat['PRN'],
                                        'Expected Spacing (deg)': 'Coverage',
                                        'Expected Position (deg)': 'N/A',
                                        'Spacing Deviation (deg)': '0.00' if coverage_health >= 100 else f"{active_ideal_regular - len(sat_angles_sorted):.2f}",
                                        'Angle (°)': f"{sat['Angle']:.2f}",
                                        'Next Spacing (°)': f"{actual_spacing:.2f}",
                                        'Position Deviation (°)': '0.00',
                                        'Health Score': f"{coverage_health:.1f}",
                                        'Health Status': health_status,
                                        'Status': '✅' if coverage_health >= 100 else '❌'
                                    })
                            elif active_spacing_strategy == 'gps_nominal':
                                # Use the occupancy map as single source of truth.
                                # Includes baseline + expansion sats in the ring pattern.
                                classified_records = []
                                if gps_occupancy is not None:
                                    classified_records = (
                                        list(gps_occupancy['baseline'])
                                        + list(gps_occupancy['expansion'])
                                    )
                                classified_sorted = sorted(
                                    classified_records, key=lambda r: r['actual_angle']
                                )

                                # Rebuild sat_angles_sorted from classified set so downstream
                                # metrics/expanders see the expansion sats too.
                                sat_angles_sorted = [
                                    {
                                        'SVN': rec['svn'],
                                        'Slot': rec['raw_slot'],
                                        'PRN': rec['prn'],
                                        'Angle': rec['actual_angle'],
                                        'Type': 'Baseline' if rec in gps_occupancy['baseline'] else 'Expansion',
                                    }
                                    for rec in classified_sorted
                                ]

                                actual_spacings_list = []
                                for i, rec in enumerate(classified_sorted):
                                    nxt = classified_sorted[(i + 1) % len(classified_sorted)]
                                    actual_spacings_list.append(
                                        (nxt['actual_angle'] - rec['actual_angle']) % 360
                                    )

                                # Symmetry uses baseline-only spacings, since expansion is
                                # per-plane structural and not symmetric across pairs.
                                baseline_sorted = sorted(
                                    gps_occupancy['baseline'] if gps_occupancy else [],
                                    key=lambda r: r['actual_angle'],
                                )
                                if baseline_sorted:
                                    baseline_spacings = [
                                        (baseline_sorted[(i + 1) % len(baseline_sorted)]['actual_angle']
                                         - baseline_sorted[i]['actual_angle']) % 360
                                        for i in range(len(baseline_sorted))
                                    ]
                                    plane_spacings_for_symmetry[plane_num] = sorted(baseline_spacings)

                                effective_nominal = (
                                    gps_occupancy['effective_nominal_spacings']
                                    if gps_occupancy else []
                                )

                                gps_pattern_result = analyze_gps_spacing_pattern(
                                    actual_spacings_list,
                                    plane_num,
                                    explicit_nominal=effective_nominal or None,
                                )

                                for i, rec in enumerate(classified_sorted):
                                    nxt = classified_sorted[(i + 1) % len(classified_sorted)]
                                    actual_spacing = (nxt['actual_angle'] - rec['actual_angle']) % 360
                                    expected_spacing = (
                                        (nxt['expected_angle'] - rec['expected_angle']) % 360
                                    )
                                    deviation = (actual_spacing - expected_spacing + 180) % 360 - 180
                                    position_deviation = rec['deviation']

                                    # Same Gaussian formula for baseline and expansion (choice A).
                                    spacing_score = 100 * np.exp(-0.004 * (deviation ** 2))
                                    position_score = 100 * np.exp(-0.004 * (position_deviation ** 2))
                                    health_score = 0.6 * spacing_score + 0.4 * position_score

                                    if health_score >= 95:
                                        health_status = '🟢 Excellent'
                                    elif health_score >= 85:
                                        health_status = '🟡 Good'
                                    elif health_score >= 70:
                                        health_status = '🟠 Fair'
                                    else:
                                        health_status = '🔴 Poor'

                                    spacing_data.append({
                                        'Slot': rec['raw_slot'],
                                        'Position': rec['assigned_label'],
                                        'SVN': rec['svn'],
                                        'PRN': rec['prn'],
                                        'Expected Spacing (deg)': f"{expected_spacing:.2f}",
                                        'Expected Position (deg)': f"{rec['expected_angle']:.2f}",
                                        'Spacing Deviation (deg)': f"{deviation:.2f}",
                                        'Angle (°)': f"{rec['actual_angle']:.2f}",
                                        'Next Spacing (°)': f"{actual_spacing:.2f}",
                                        'Position Deviation (°)': f"{position_deviation:.2f}",
                                        'Health Score': f"{health_score:.1f}",
                                        'Health Status': health_status,
                                        'Status': '✅' if abs(deviation) <= 5 else '⚠️' if abs(deviation) <= 12 else '❌',
                                    })
                            else:
                                nominal_spacings = list(active_nominal_spacings) or [360 / len(sat_angles_sorted) if len(sat_angles_sorted) > 0 else 45]
                                expected_positions = assign_expected_positions_by_geometry(sat_angles_sorted, ideal_angles)

                                for i, sat in enumerate(sat_angles_sorted):
                                    next_sat = sat_angles_sorted[(i + 1) % len(sat_angles_sorted)]
                                    actual_spacing = (next_sat['Angle'] - sat['Angle']) % 360
                                    expected_spacing = (
                                        nominal_spacings[i % len(nominal_spacings)]
                                        if len(sat_angles_sorted) == len(nominal_spacings)
                                        else min(nominal_spacings, key=lambda x: abs(actual_spacing - x))
                                    )

                                    deviation_45 = actual_spacing - expected_spacing
                                    if deviation_45 > 180:
                                        deviation_45 -= 360
                                    elif deviation_45 < -180:
                                        deviation_45 += 360

                                    expected_position = expected_positions.get(sat['SVN'])
                                    if expected_position is None:
                                        expected_position = min(
                                            ideal_angles,
                                            key=lambda x: circular_distance_degrees(sat['Angle'], x)
                                        )
                                    position_deviation = circular_delta_degrees(sat['Angle'], expected_position)

                                    spacing_score = 100 * np.exp(-0.01 * (deviation_45 ** 2))
                                    position_score = 100 * np.exp(-0.01 * (position_deviation ** 2))
                                    health_score = 0.6 * spacing_score + 0.4 * position_score

                                    if health_score >= 95:
                                        health_status = '🟢 Excellent'
                                    elif health_score >= 85:
                                        health_status = '🟡 Good'
                                    elif health_score >= 70:
                                        health_status = '🟠 Fair'
                                    else:
                                        health_status = '🔴 Poor'

                                    spacing_data.append({
                                        'Slot': sat['Slot'],
                                        'SVN': sat['SVN'],
                                        'PRN': sat['PRN'],
                                        'Expected Spacing (deg)': f"{expected_spacing:.2f}",
                                        'Expected Position (deg)': f"{expected_position:.2f}",
                                        'Spacing Deviation (deg)': f"{deviation_45:.2f}",
                                        'Angle (°)': f"{sat['Angle']:.2f}",
                                        'Next Spacing (°)': f"{actual_spacing:.2f}",
                                        'Deviation from 45° (°)': f"{deviation_45:.2f}",
                                        'Position Deviation (°)': f"{position_deviation:.2f}",
                                        'Health Score': f"{health_score:.1f}",
                                        'Health Status': health_status,
                                        'Status': '✅' if abs(deviation_45) <= 2 else '⚠️' if abs(deviation_45) <= 5 else '❌'
                                    })
                            
                            spacing_df = pd.DataFrame(spacing_data)
                            legacy_deviation_cols = [
                                col for col in spacing_df.columns
                                if str(col).startswith('Deviation from 45')
                            ]
                            if legacy_deviation_cols:
                                spacing_df = spacing_df.drop(columns=legacy_deviation_cols)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                sats_label = (
                                    "Classified Sats" if active_spacing_strategy == 'gps_nominal'
                                    else "Regular Sats"
                                )
                                st.metric(sats_label, len(sat_angles_sorted))
                            with col2:
                                good_spacing = len(spacing_df[spacing_df['Status'] == '✅'])
                                spacing_metric_label = (
                                    "Coverage OK" if active_spacing_strategy == 'coverage'
                                    else "Within ±5°" if active_spacing_strategy == 'gps_nominal'
                                    else "Within ±2°"
                                )
                                st.metric(spacing_metric_label, f"{good_spacing}/{len(spacing_df)}")
                            with col3:
                                if len(spacing_df) > 0:
                                    avg_health = pd.to_numeric(spacing_df['Health Score']).mean()
                                    st.metric("Avg Health", f"{avg_health:.1f}/100")
                                else:
                                    st.metric("Avg Health", "N/A")
                            with col4:
                                avg_deviation = abs(pd.to_numeric(spacing_df['Spacing Deviation (deg)'])).mean()
                                if active_spacing_strategy == 'coverage':
                                    st.metric("Baseline Gap", f"{avg_deviation:.0f}")
                                elif active_spacing_strategy == 'gps_nominal':
                                    st.metric("Avg Dev (GPS)", f"{avg_deviation:.2f}°")
                                else:
                                    st.metric("Avg Deviation", f"{avg_deviation:.2f}°")
                            
                            if len(spacing_df) > 0:
                                st.dataframe(spacing_df, hide_index=True, use_container_width=True)
                            else:
                                st.warning("No regular satellites found in this plane")
                            
                            aux_sats = plot_df[plot_df['Type'] == 'Auxiliary']
                            spare_sats = plot_df[plot_df['Type'] == 'Spare']
                            if len(aux_sats) > 0:
                                if active_name == 'GPS' and 'GPS_Assigned_Label' in aux_sats.columns:
                                    labels = [
                                        f"{row['GPS_Assigned_Label']} [{row['Slot']}]"
                                        for _, row in aux_sats.iterrows()
                                    ]
                                    st.caption(f"**Expansion Satellites:** {', '.join(labels)}")
                                else:
                                    labels = [
                                        f"{row['Slot']} (filling {row['Baseline_Fill_Label']})"
                                        if pd.notna(row['Baseline_Fill_Label']) else row['Slot']
                                        for _, row in aux_sats.iterrows()
                                    ]
                                    st.caption(f"**Auxiliary Satellites:** {', '.join(labels)}")
                            if active_name == 'GPS' and gps_slot_summary:
                                state_bits = [
                                    f"{label} {GPS_SLOT_STATE_LABELS.get(state, state)}"
                                    for label, state in gps_slot_summary['slot_states'].items()
                                ]
                                st.caption(
                                    f"**Slots ({gps_slot_summary['slots_occupied']}"
                                    f"/{gps_slot_summary['slots_total']} occupied):** "
                                    + " · ".join(state_bits)
                                )
                                if gps_slot_summary['degraded_slots']:
                                    st.warning(
                                        f"⚠️ {', '.join(gps_slot_summary['degraded_slots'])} holds only "
                                        f"one half of its expanded pair. Per SPS PS 3.7.1 note 3 the slot "
                                        f"has lost availability until the missing satellite is replaced or "
                                        f"the slot is returned to its baseline configuration."
                                    )
                            if len(spare_sats) > 0:
                                if active_name == 'GPS':
                                    aux_bits = []
                                    for aux in (gps_slot_summary or {}).get('auxiliaries', []):
                                        if aux['shadows']:
                                            aux_bits.append(
                                                f"{aux['prn']} shadowing {aux['shadows']} "
                                                f"({aux['offset_deg']:+.2f}°)"
                                            )
                                        else:
                                            aux_bits.append(f"{aux['prn']} (unattached)")
                                    st.caption(
                                        "**Auxiliary satellites** — SPS PS 3.2 gives auxiliaries no "
                                        "a priori slot; one parked beside an occupied position is "
                                        "normally a replacement in waiting: "
                                        + ", ".join(aux_bits or spare_sats['PRN'].tolist())
                                    )
                                else:
                                    labels = [
                                        f"{row['PRN']} (filling {row['Baseline_Fill_Label']})"
                                        if pd.notna(row['Baseline_Fill_Label']) else row['PRN']
                                        for _, row in spare_sats.iterrows()
                                    ]
                                    st.caption(f"**Spare Satellites:** {', '.join(labels)}")
                        
                        # GPS-specific: Spacing Pattern Analysis
                        if active_spacing_strategy == 'gps_nominal' and gps_pattern_result:
                            with st.expander("📐 Spacing Pattern Analysis", expanded=False):
                                pattern = gps_pattern_result
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    tp_icon = "✅" if pattern['tight_pair_found'] else "❌"
                                    tp_val = f"{pattern['tight_pair_value']:.1f}°" if pattern['tight_pair_found'] else "None"
                                    st.metric(f"{tp_icon} Tight Pair", tp_val, f"Expected {GPS_TIGHT_PAIR_RANGE[0]}-{GPS_TIGHT_PAIR_RANGE[1]}°")
                                with col2:
                                    mg_icon = "✅" if pattern['max_gap_status'] == 'ok' else "⚠️" if pattern['max_gap_status'] == 'warning' else "❌"
                                    st.metric(f"{mg_icon} Max Gap", f"{pattern['max_gap']:.1f}°", f"Warn > {GPS_MAX_GAP_WARNING}°")
                                with col3:
                                    st.metric("Pattern RMSE", f"{pattern['rmse']:.2f}°", f"Score: {pattern['pattern_score']:.0f}/100")
                        
                        # GPS-specific: Expansion Slot Classification
                        if active_spacing_strategy == 'gps_nominal' and gps_occupancy is not None:
                            with st.expander("📎 Expansion Slot Classification", expanded=False):
                                expandable = gps_occupancy.get('expandable_slot')
                                if expandable:
                                    st.caption(
                                        f"Plane's expandable baseline slot: **{expandable}**. "
                                        f"Aft/fore expansion satellites are classified by signed "
                                        f"offset from that slot."
                                    )
                                if gps_occupancy['expansion']:
                                    exp_rows = []
                                    for rec in gps_occupancy['expansion']:
                                        icon = '✅' if abs(rec['deviation']) <= 5 else '⚠️' if abs(rec['deviation']) <= GPS_EXPANSION_CLASSIFY_TOL else '❌'
                                        exp_rows.append({
                                            'Status': icon,
                                            'SVN': rec['svn'],
                                            'PRN': rec['prn'],
                                            'Raw Slot (JSON)': rec['raw_slot'],
                                            'Assigned Position': rec['assigned_label'],
                                            'Direction': rec['position'].title(),
                                            'Expected Offset (°)': f"{rec['expected_offset']:+.2f}",
                                            'Actual Offset (°)': f"{rec['actual_offset']:+.2f}",
                                            'Deviation (°)': f"{rec['deviation']:+.2f}",
                                        })
                                    st.dataframe(pd.DataFrame(exp_rows), hide_index=True, use_container_width=True)
                                else:
                                    st.info("No expansion satellites classified in this plane.")

                                vacancies_msgs = []
                                if gps_occupancy['baseline_vacancies']:
                                    vacancies_msgs.append(
                                        f"Baseline vacancies: {', '.join(gps_occupancy['baseline_vacancies'])}"
                                    )
                                if gps_occupancy['expansion_vacancies']:
                                    vacancies_msgs.append(
                                        f"Expansion vacancies: {', '.join(gps_occupancy['expansion_vacancies'])}"
                                    )
                                if vacancies_msgs:
                                    st.caption(" | ".join(vacancies_msgs))

                                if gps_occupancy['unclassified']:
                                    unc_rows = []
                                    for rec in gps_occupancy['unclassified']:
                                        unc_rows.append({
                                            'SVN': rec['svn'],
                                            'PRN': rec['prn'],
                                            'Raw Slot (JSON)': rec['raw_slot'],
                                            'Actual Angle (°)': f"{rec['actual_angle']:.2f}",
                                            'Nearest Expected': rec['nearest_label'],
                                            'Deviation (°)': f"{rec['nearest_deviation']:+.2f}",
                                        })
                                    st.warning("Unclassified satellites (outside baseline/expansion tolerance):")
                                    st.dataframe(pd.DataFrame(unc_rows), hide_index=True, use_container_width=True)
                        
                        # Plane age health (youngest-N + spare bonus) — computed once,
                        # consumed by the Age Analysis expander and the plane health score.
                        plane_launch_dates = st.session_state.get(launch_dates_key, {})
                        plane_age_entries = []
                        for svn in plane_df['SVN'].tolist():
                            if svn in plane_launch_dates:
                                age_years = (STATIC_ANALYSIS_EPOCH - plane_launch_dates[svn]).days / 365.25
                                plane_age_entries.append({
                                    'svn': svn,
                                    'age_ratio': age_years / active_design_life,
                                    'score': calculate_satellite_age_score(
                                        plane_launch_dates[svn], active_design_life),
                                })
                        plane_age_health = (
                            calculate_plane_age_health(plane_age_entries, active_ideal_regular)
                            if plane_launch_dates else None
                        )

                        # Age analysis expander
                        with st.expander("📅 Satellite Age Analysis", expanded=False):
                            launch_dates = st.session_state.get(launch_dates_key, {})
                            
                            if not launch_dates:
                                st.info("📅 Launch date data not available. Launch dates are unavailable in this snapshot.")
                            else:
                                st.caption("**Age and design life assessment for satellites in this plane**")
                                st.caption(
                                    f"Scoring: plane age health = mean of the youngest "
                                    f"{active_ideal_regular} satellites (20 pts) + 2.5 pts per "
                                    f"additional usable satellite within design life (max 5 pts). "
                                    f"Older retained satellites are not penalized."
                                )

                                age_data = []
                                plane_svn_list = plane_df['SVN'].tolist()
                                
                                for svn in plane_svn_list:
                                    if svn in launch_dates:
                                        launch_date = launch_dates[svn]
                                        age_years = (STATIC_ANALYSIS_EPOCH - launch_date).days / 365.25
                                        design_life = active_design_life
                                        age_ratio = age_years / design_life
                                        sat_age_score = calculate_satellite_age_score(launch_date, design_life)
                                        
                                        if age_ratio < 0.8:
                                            age_status = "🟢 Good"
                                        elif age_ratio < 1.0:
                                            age_status = "🟡 Approaching"
                                        elif age_ratio < 1.2:
                                            age_status = "🟠 At Design Life"
                                        else:
                                            age_status = "🔴 Exceeded"
                                        
                                        age_data.append({
                                            'SVN': svn,
                                            'Slot': active_sats[svn]['slot'],
                                            'PRN': active_sats[svn]['prn'],
                                            'Launch Date': launch_date.strftime('%Y-%m-%d'),
                                            'Age (years)': f"{age_years:.1f}",
                                            'Design Life': f"{design_life} yrs",
                                            'Age %': f"{age_ratio*100:.0f}%",
                                            'Age Score': f"{sat_age_score:.0f}",
                                            'Counted As': {
                                                'scored': f'Scored (youngest-{active_ideal_regular})',
                                                'bonus': 'Spare bonus',
                                                'not_counted': 'Not counted (past design life)',
                                            }.get((plane_age_health or {}).get('counted_by_svn', {}).get(svn), '—'),
                                            'Status': age_status
                                        })
                                
                                if age_data:
                                    age_df = pd.DataFrame(age_data)
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Total Satellites", len(age_data))
                                    with col2:
                                        avg_age = np.mean([float(d['Age (years)']) for d in age_data])
                                        st.metric("Avg Age", f"{avg_age:.1f} yrs")
                                    with col3:
                                        avg_age_score_val = np.mean([float(d['Age Score']) for d in age_data])
                                        st.metric("Avg Age Score", f"{avg_age_score_val:.0f}/100")
                                    with col4:
                                        old_sats = len([d for d in age_data if float(d['Age %'].replace('%', '')) >= 100])
                                        st.metric("At/Exceed Design Life", f"{old_sats}/{len(age_data)}")
                                    
                                    st.dataframe(age_df, hide_index=True, use_container_width=True)
                                    
                                    if old_sats > 0:
                                        st.warning(f"⚠️ {old_sats} satellite(s) at or exceeding design life")
                                else:
                                    available_svns = [svn for svn in plane_svn_list if svn in launch_dates]
                                    if available_svns:
                                        st.info(f"Launch dates found for: {', '.join(available_svns)}")
                                    else:
                                        st.info(f"No launch date data for satellites in this plane ({', '.join(plane_svn_list)})")
                        
                        
                        # Calculate overall plane health score
                        st.markdown("---")
                        
                        regular_sats = len(plot_df[plot_df['Type'] == 'Regular'])
                        aux_sats_count = len(plot_df[plot_df['Type'] == 'Auxiliary'])
                        spare_sats_count = len(plot_df[plot_df['Type'] == 'Spare'])
                        reserve_sats_count = aux_sats_count + spare_sats_count
                        satellite_mix_value, satellite_mix_delta = format_satellite_mix(
                            regular_sats,
                            reserve_sats_count
                        )
                        ideal_regular = active_ideal_regular
                        ideal_auxiliary = active_ideal_auxiliary

                        # GPS is scored on SLOTS occupied, not on satellites
                        # sitting at baseline centres. An expanded slot has an
                        # empty centre and a fore/aft pair, so counting centre
                        # occupants marked a fully-populated expanded plane as
                        # 3/4 -- penalising it for a configuration the spec
                        # defines as equivalent-or-better (SPS PS A.7.2.4/A.7.2.5).
                        if active_name == 'GPS' and gps_slot_summary:
                            occupied_units = gps_slot_summary['slots_occupied']
                            ideal_units = gps_slot_summary['slots_total'] or ideal_regular
                            satellite_mix_value = (
                                f"{occupied_units}/{ideal_units} slots"
                            )
                            satellite_mix_delta = (
                                f"{len(plot_df)} SV"
                                + (f" · {len(gps_slot_summary['auxiliaries'])} aux"
                                   if gps_slot_summary['auxiliaries'] else "")
                            )
                        else:
                            occupied_units = regular_sats
                            ideal_units = ideal_regular

                        regular_score = min(20, (occupied_units / ideal_units) * 20) if ideal_units > 0 else 20
                        aux_score = min(5, (reserve_sats_count / ideal_auxiliary) * 5) if ideal_auxiliary > 0 else 5
                        satellite_count_score = regular_score + aux_score
                        
                        if len(spacing_df) > 0:
                            avg_satellite_health = pd.to_numeric(spacing_df['Health Score']).mean()
                            spacing_quality_score = (avg_satellite_health / 100) * 50
                        else:
                            avg_satellite_health = 0
                            spacing_quality_score = 0
                        
                        if plane_age_health is not None:
                            avg_age_score = plane_age_health['youngest_mean']
                            age_base_points = plane_age_health['age_base_points']
                            spare_bonus_points = plane_age_health['spare_bonus_points']
                            age_score = plane_age_health['age_points']
                        else:
                            avg_age_score = 50
                            age_base_points = 12.5
                            spare_bonus_points = 0.0
                            age_score = 12.5
                        
                        plane_health_score = satellite_count_score + spacing_quality_score + age_score
                        
                        if plane_health_score >= 90:
                            plane_health_status = '🟢 Excellent'
                        elif plane_health_score >= 75:
                            plane_health_status = '🟡 Good'
                        elif plane_health_score >= 60:
                            plane_health_status = '🟠 Fair'
                        else:
                            plane_health_status = '🔴 Poor'
                        
                        plane_health_data[plane_num] = {
                            'score': plane_health_score,
                            'status': plane_health_status,
                            'regular_sats': regular_sats,
                            'aux_sats': aux_sats_count,
                            'spare_sats': spare_sats_count,
                            'reserve_sats': reserve_sats_count,
                            'satellite_mix_value': satellite_mix_value,
                            'satellite_mix_delta': satellite_mix_delta,
                            'avg_sat_health': avg_satellite_health,
                            'avg_age_score': avg_age_score,
                            'age_base_points': age_base_points,
                            'spare_bonus_points': spare_bonus_points,
                            'spacing_quality_score': spacing_quality_score,
                        }
                        
                        st.markdown("#### 📊 Plane Health Score")
                        
                        health_color = "green" if plane_health_score >= 90 else "orange" if plane_health_score >= 75 else "red"
                        st.markdown(f"""
                        <div style="padding: 20px; background-color: rgba(0,0,0,0.2); border-radius: 10px; border-left: 5px solid {health_color};">
                            <h2 style="margin: 0; color: white;">{plane_health_score:.0f}/100 {plane_health_status}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("")  # Spacing
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("🛰️ Satellites", satellite_mix_value, satellite_mix_delta)
                        with col2:
                            st.metric("🎯 Target", format_target_mix(ideal_regular, ideal_auxiliary))
                        with col3:
                            st.metric("💯 Spacing", f"{avg_satellite_health:.0f}")
                        with col4:
                            st.metric("📅 Age Health", f"{avg_age_score:.0f}")
                        with col5:
                            st.metric("🌍 RAAN", f"{plane_mean_raans.get(plane_num, 0):.1f}°")
                        st.caption(
                            f"📅 Age contribution: {age_base_points:.1f}/20 "
                            f"(youngest-{active_ideal_regular} mean {avg_age_score:.0f}/100) "
                            f"+ {spare_bonus_points:.1f}/5 spare bonus"
                        )
            
            # Display constellation health summary at the top (using the placeholder)
            with health_summary_placeholder:
                st.markdown(f"## 🏥 {active_name} Constellation Health Summary")
                st.caption(f"Overall health assessment across all {active_plane_count} orbital planes")
                
                if plane_health_data:
                    if len(plane_health_data) == active_plane_count:
                        avg_constellation_health = sum(p['score'] for p in plane_health_data.values()) / active_plane_count
                        if avg_constellation_health >= 90:
                            status = '🟢 Excellent'
                            color = 'green'
                        elif avg_constellation_health >= 75:
                            status = '🟡 Good'
                            color = 'orange'
                        elif avg_constellation_health >= 60:
                            status = '🟠 Fair'
                            color = 'orange'
                        else:
                            status = '🔴 Poor'
                            color = 'red'
                        
                        st.markdown(f"""
                        <div style="padding: 30px; background: linear-gradient(135deg, rgba(0,0,0,0.3), rgba(0,0,0,0.1)); border-radius: 15px; border: 2px solid {color}; text-align: center; margin: 20px 0;">
                            <h1 style="margin: 0; color: white; font-size: 3em;">{avg_constellation_health:.0f}/100</h1>
                            <h3 style="margin: 10px 0 0 0; color: white;">{status}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("### Individual Plane Summary")
                    summary_cols = st.columns(min(active_plane_count, 3))
                    
                    for idx, plane_num in enumerate(active_plane_numbers):
                        if plane_num in plane_health_data:
                            health = plane_health_data[plane_num]
                            health_color = "green" if health['score'] >= 90 else "orange" if health['score'] >= 75 else "red"
                            
                            with summary_cols[idx % len(summary_cols)]:
                                st.markdown(f"""
                                <div style="padding: 15px; background-color: rgba(0,0,0,0.2); border-radius: 10px; border-left: 4px solid {health_color}; margin-bottom: 10px;">
                                    <h3 style="margin: 0 0 10px 0; color: white;">Plane {plane_num}</h3>
                                    <p style="margin: 5px 0; color: #888;">RAAN: {plane_mean_raans.get(plane_num, 0):.1f}°</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.metric("Health Score", f"{health['score']:.0f}/100", health['status'])
                                st.metric(
                                    "Satellites",
                                    health.get('satellite_mix_value', f"{health['regular_sats']}+{health['reserve_sats']}"),
                                    health.get('satellite_mix_delta', "Regular+Reserve")
                                )
                                st.metric("Spacing Health", f"{health['avg_sat_health']:.0f}/100")
                                st.metric("Age Health", f"{health['avg_age_score']:.0f}/100")
                
                st.markdown("---")
                st.markdown("<br>", unsafe_allow_html=True)

            # GPS-specific: Symmetry modifier pass (after all planes processed)
            if active_spacing_strategy == 'gps_nominal' and len(plane_spacings_for_symmetry) >= 2:
                opposite_map = cfg.get('opposite_planes', {})
                adjusted_planes = set()
                for plane_num in list(plane_spacings_for_symmetry.keys()):
                    opp = opposite_map.get(plane_num)
                    if opp and opp in plane_spacings_for_symmetry and plane_num not in adjusted_planes:
                        adjusted_planes.add(plane_num)
                        adjusted_planes.add(opp)
                        a_spacings = plane_spacings_for_symmetry[plane_num]
                        b_spacings = plane_spacings_for_symmetry[opp]
                        min_len = min(len(a_spacings), len(b_spacings))
                        if min_len > 0:
                            sym_dev = np.mean([
                                abs(a_spacings[i] - b_spacings[i]) for i in range(min_len)
                            ])
                            symmetry_score = 100.0 * np.exp(-0.02 * (sym_dev ** 2))
                            for pn in (plane_num, opp):
                                if pn in plane_health_data:
                                    old_sq = plane_health_data[pn].get('spacing_quality_score', 50)
                                    blended = old_sq * 0.9 + (symmetry_score / 100) * 50 * 0.1
                                    plane_health_data[pn]['spacing_quality_score'] = blended
                                    old_score = plane_health_data[pn]['score']
                                    new_score = old_score - old_sq + blended
                                    plane_health_data[pn]['score'] = new_score
                                    plane_health_data[pn]['symmetry_score'] = symmetry_score
                                    if new_score >= 90:
                                        plane_health_data[pn]['status'] = '🟢 Excellent'
                                    elif new_score >= 75:
                                        plane_health_data[pn]['status'] = '🟡 Good'
                                    elif new_score >= 60:
                                        plane_health_data[pn]['status'] = '🟠 Fair'
                                    else:
                                        plane_health_data[pn]['status'] = '🔴 Poor'

            # GPS-specific: Cross-plane symmetry analysis section
            if active_spacing_strategy == 'gps_nominal' and len(plane_spacings_for_symmetry) >= 2:
                st.markdown("---")
                st.markdown("## 📐 Cross-Plane Symmetry Analysis")
                st.caption("Spacing pattern mirroring between opposite plane pairs (A↔F, B↔E, C↔D)")

                symmetry_result = analyze_gps_plane_symmetry(
                    plane_spacings_for_symmetry,
                    cfg.get('opposite_planes', {})
                )

                for pair_key, pair_data in symmetry_result['pair_scores'].items():
                    plane_a, plane_b = pair_data['plane_nums']
                    with st.expander(f"Plane {plane_a} ↔ Plane {plane_b} (Pair {pair_key})", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Symmetry Score", f"{pair_data['score']:.0f}/100")
                        with col2:
                            st.metric("Mean Deviation", f"{pair_data['deviation']:.2f}°")

                        comparison_df = pd.DataFrame({
                            f'Plane {plane_a} Spacing': pair_data['spacings_a'],
                            f'Plane {plane_b} Spacing': pair_data['spacings_b'],
                            'Difference': pair_data['details'],
                        })
                        st.dataframe(comparison_df, hide_index=True, use_container_width=True)

                st.metric("Overall Symmetry Score", f"{symmetry_result['overall_score']:.0f}/100")
                st.markdown("<br>", unsafe_allow_html=True)

            if active_name == 'GPS' and legacy_assignments:
                departed = {
                    key: assignment for key, assignment in legacy_assignments.items()
                    if assignment.get('tier') == GPS_LEGACY_TIER_DEPARTED
                }
                if departed:
                    st.markdown("---")
                    st.markdown(
                        f"#### 🕐 Legacy satellites no longer plane-resident "
                        f"({len(departed)})"
                    )
                    st.caption(
                        "A RAAN offset above "
                        f"{GPS_LEGACY_RESIDENT_DEG:.0f}° means the object sits between "
                        "planes rather than in one. They are kept off the Circular "
                        "View, whose geometry assumes every plotted satellite is "
                        "coplanar."
                    )
                    departed_rows = []
                    for key, assignment in sorted(departed.items()):
                        roster_entry = GPS_LEGACY_SATS.get(key, {})
                        state = legacy_states.get(key) or {}
                        departed_rows.append({
                            'Name': roster_entry.get('name', key),
                            'NORAD': roster_entry.get('norad', 0),
                            'SATCAT status': roster_entry.get('satcat_status', 'unknown'),
                            'Launched': roster_entry.get('launch_date', ''),
                            'Nearest plane': GPS_PLANE_LETTERS.get(
                                assignment['nearest_plane'], '—'),
                            'RAAN offset (°)': round(assignment['raan_offset_deg'], 2),
                            'Inclination (°)': round(state.get('inclination', float('nan')), 2),
                        })
                    st.dataframe(pd.DataFrame(departed_rows),
                                 use_container_width=True, hide_index=True)

        with tabs[tab_idx]:  # Overview (now second)
            tab_idx += 1
            st.markdown(f"## 📊 {active_name} Constellation Status Overview")
            st.caption("Detailed drift analysis and operational status by orbital plane")
            
            for plane_num in active_plane_numbers:
                plane_df = df[df['Plane'] == f"Plane {plane_num}"]
                mean_raan = plane_mean_raans.get(plane_num, active_nominal_raan.get(plane_num, 0))
                operational_count = len(plane_df[plane_df['Status'] == 'USABLE'])
                
                st.markdown("---")
                
                if plane_num in plane_health_data:
                    health = plane_health_data[plane_num]
                    health_color = "green" if health['score'] >= 90 else "orange" if health['score'] >= 75 else "red"
                    
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; margin-bottom: 15px;">
                        <h2 style="margin: 0; color: white;">🛰️ Plane {plane_num}</h2>
                        <div style="margin-left: 20px; padding: 8px 15px; background-color: rgba(0,0,0,0.3); border-radius: 20px; border: 2px solid {health_color};">
                            <span style="color: white; font-weight: bold;">{health['score']:.0f}/100 {health['status']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"## 🛰️ Plane {plane_num}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🟢 Operational", f"{operational_count}/{len(plane_df)}")
                with col2:
                    st.metric("🌍 Mean RAAN", f"{mean_raan:.1f}°")
                with col3:
                    in_plane_ok = len(plane_df[plane_df['In-Plane OK'] == '✅'])
                    st.metric("✅ In-Plane OK", f"{in_plane_ok}/{len(plane_df)}")
                with col4:
                    across_ok = len(plane_df[plane_df['Across-Track OK'] == '✅'])
                    st.metric("✅ Across-Track OK", f"{across_ok}/{len(plane_df)}")
                
                st.markdown("")
                
                plane_tab1, plane_tab2 = st.tabs(["📈 In-Plane Analysis", "🌍 Across-Track Analysis"])
                
                with plane_tab1:
                    st.dataframe(
                        plane_df[[
                            'SVN', 'PRN', 'Slot', 'Type',
                            'TLE Epoch (UTC)', 'Δ Common Epoch (days)', '|Δ Common Epoch| (days)',
                            'Argument of Latitude (°)', 'In-Plane Drift (km/day)', 'In-Plane OK'
                        ]],
                        hide_index=True,
                        use_container_width=True
                    )
                
                with plane_tab2:
                    st.dataframe(
                        plane_df[[
                            'SVN', 'PRN', 'Slot', 'Type',
                            'TLE Epoch (UTC)', 'Δ Common Epoch (days)', '|Δ Common Epoch| (days)',
                            'Cross-Track (km)', 'Across-Track OK'
                        ]],
                        hide_index=True,
                        use_container_width=True
                    )
        
        with tabs[tab_idx]:  # In-Plane Drift
            tab_idx += 1
            st.markdown(f"## 📈 {active_name} In-Plane Drift Analysis")
            st.caption("Along-track position deviation monitoring (±2° tolerance)")
            
            operational_df = df[df['Plane'] != "Non-nominal"]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                in_tolerance = len(operational_df[operational_df['In-Plane OK'] == '✅'])
                st.metric("✅ Within Tolerance", f"{in_tolerance}/{len(operational_df)}")
            with col2:
                avg_drift = operational_df['In-Plane Drift (km/day)'].mean()
                st.metric("📊 Avg Drift", f"{avg_drift:.2f} km/day")
            with col3:
                max_drift = operational_df['In-Plane Drift (km/day)'].abs().max()
                st.metric("⚠️ Max Drift", f"{max_drift:.2f} km/day")
            with col4:
                tolerance = 2 * active_km_per_deg
                st.metric("🎯 Tolerance", f"±{tolerance:.1f} km/day")
            
            st.markdown("---")
            
            st.markdown("### 📍 Drift by Satellite")
            fig = px.scatter(operational_df, x='SVN', y='In-Plane Drift (km/day)', 
                           color='Plane', size='Cross-Track (km)',
                           hover_data=['PRN', 'Slot', 'Status', 'Mean Motion (rev/day)', 'Argument of Latitude (°)'],
                           title=f"{active_name} In-Plane Drift by Satellite")
            fig.add_hline(y=2*active_km_per_deg, line_dash="dash", line_color="red", 
                         annotation_text="Upper Tolerance (+2°)")
            fig.add_hline(y=-2*active_km_per_deg, line_dash="dash", line_color="red",
                         annotation_text="Lower Tolerance (-2°)")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("")
            
            st.markdown("### 📊 Distribution Analysis")
            fig2 = px.histogram(operational_df, x='In-Plane Drift (km/day)', nbins=30,
                              title=f"Distribution of {active_name} In-Plane Drift",
                              labels={'In-Plane Drift (km/day)': 'Drift (km/day)'})
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        with tabs[tab_idx]:  # Across-Track Drift
            tab_idx += 1
            st.markdown(f"## 🌍 {active_name} Across-Track Drift Analysis")
            st.caption("Orbital plane deviation monitoring (±2° RAAN tolerance)")
            
            operational_df = df[df['Plane'] != "Non-nominal"]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                across_ok = len(operational_df[operational_df['Across-Track OK'] == '✅'])
                st.metric("✅ Within Tolerance", f"{across_ok}/{len(operational_df)}")
            with col2:
                avg_raan_dev = operational_df['ΔRAAN (°)'].abs().mean()
                st.metric("📊 Avg RAAN Dev", f"{avg_raan_dev:.2f}°")
            with col3:
                avg_cross_track = operational_df['Cross-Track (km)'].abs().mean()
                st.metric("📏 Avg Cross-Track", f"{avg_cross_track:.1f} km")
            with col4:
                st.metric("🎯 Tolerance", "±2°")
            
            st.markdown("---")
            
            st.markdown("### 📍 RAAN Deviation by Satellite")
            fig = px.scatter(operational_df, x='SVN', y='ΔRAAN (°)', 
                           color='Plane', size='Cross-Track (km)',
                           hover_data=['PRN', 'Slot', 'Status', 'RAAN (°)'],
                           title=f"{active_name} RAAN Deviation by Satellite")
            fig.add_hline(y=2, line_dash="dash", line_color="red", 
                         annotation_text="Upper Tolerance (+2°)")
            fig.add_hline(y=-2, line_dash="dash", line_color="red",
                         annotation_text="Lower Tolerance (-2°)")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("")
            
            st.markdown("### 📏 Cross-Track Displacement")
            fig2 = px.bar(operational_df, x='SVN', y='Cross-Track (km)', 
                         color='Plane',
                         hover_data=['PRN', 'Slot', 'Status'],
                         title=f"{active_name} Cross-Track Displacement (km)")
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        with tabs[tab_idx]:  # DOP Calculator
            tab_idx += 1
            st.markdown("## 📡 Dilution of Precision (DOP) Calculator")
            st.caption("Calculate positioning accuracy for any location on Earth")
            st.caption(f"Using common SGP4 analysis epoch: {analysis_epoch.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            
            # Modern info box
            st.markdown("""
            <div style="padding: 25px; background: linear-gradient(135deg, rgba(0,100,200,0.15), rgba(0,150,100,0.1)); border-radius: 12px; margin: 20px 0;">
                <h4 style="color: #4CAF50; margin-bottom: 15px;">📊 Understanding DOP Values</h4>
                <p style="color: #ddd; margin-bottom: 10px;"><strong>DOP (Dilution of Precision)</strong> measures the geometric quality of satellite positioning. Lower values indicate better accuracy.</p>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 15px;">
                    <div style="padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px;">
                        <strong style="color: #64B5F6;">GDOP</strong> - Geometric (Overall 3D + time)
                    </div>
                    <div style="padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px;">
                        <strong style="color: #81C784;">PDOP</strong> - Position (3D accuracy)
                    </div>
                    <div style="padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px;">
                        <strong style="color: #FFB74D;">HDOP</strong> - Horizontal (Lat/Lon)
                    </div>
                    <div style="padding: 10px; background: rgba(255,255,255,0.05); border-radius: 8px;">
                        <strong style="color: #F06292;">VDOP</strong> - Vertical (Altitude)
                    </div>
                </div>
                <div style="margin-top: 15px; padding: 12px; background: rgba(255,255,255,0.08); border-radius: 8px;">
                    <p style="margin: 0; color: #aaa; font-size: 0.95em;"><strong>Quality Guide:</strong> 
                    <span style="color: #00E676;">&lt;1: Ideal ⭐</span> | 
                    <span style="color: #4CAF50;">1-2: Excellent 🟢</span> | 
                    <span style="color: #FFEB3B;">2-5: Good 🟡</span> | 
                    <span style="color: #FF9800;">5-10: Moderate 🟠</span> | 
                    <span style="color: #F44336;">10-20: Fair 🔴</span> | 
                    <span style="color: #9E9E9E;">&gt;20: Poor ⚫</span></p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")
            
            # Combined DOP option
            other_constellations = [name for name in CONSTELLATION_CONFIG if name != constellation_choice]
            available_combined_constellations = [
                name for name in other_constellations
                if f'tle_data_{name}' in st.session_state
            ]
            missing_combined_constellations = [
                name for name in other_constellations
                if f'tle_data_{name}' not in st.session_state
            ]

            selected_combined_constellations = []
            if available_combined_constellations:
                selected_combined_constellations = st.multiselect(
                    "Include additional constellations for combined DOP",
                    options=available_combined_constellations,
                    default=[],
                    help=f"Merge {constellation_choice} with other loaded constellations for multi-constellation DOP calculation"
                )
            if missing_combined_constellations:
                st.info(
                    "The snapshot lacks orbital data for: "
                    + ", ".join(missing_combined_constellations)
                )

            use_combined = bool(selected_combined_constellations)
            dop_mode_label = " + ".join([constellation_choice] + selected_combined_constellations) if use_combined else constellation_choice

            for key, default_value in {
                'dop_latitude': 0.0,
                'dop_longitude': 0.0,
                'dop_altitude': 0.0,
                'dop_elevation_mask': 5.0,
            }.items():
                if key not in st.session_state:
                    st.session_state[key] = default_value

            pending_map_location = st.session_state.pop('dop_pending_map_location', None)
            if pending_map_location is not None:
                st.session_state['dop_latitude'] = pending_map_location['lat']
                st.session_state['dop_longitude'] = pending_map_location['lon']
                st.session_state['dop_auto_calculate'] = True

            # Location input section
            st.markdown("""
            <div style="padding: 15px; background: linear-gradient(90deg, rgba(100,181,246,0.2), rgba(66,165,245,0.1)); border-radius: 10px; margin: 20px 0 15px 0;">
                <h3 style="margin: 0; color: white;">📍 Observer Location</h3>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                latitude = st.number_input(
                    "Latitude (deg)",
                    min_value=-90.0, 
                    max_value=90.0, 
                    step=0.1,
                    key='dop_latitude',
                    help="Enter latitude in degrees (-90 to 90)"
                )
            
            with col2:
                longitude = st.number_input(
                    "Longitude (deg)",
                    min_value=-180.0, 
                    max_value=180.0, 
                    step=0.1,
                    key='dop_longitude',
                    help="Enter longitude in degrees (-180 to 180)"
                )
            
            with col3:
                altitude = st.number_input(
                    "Altitude (km)", 
                    min_value=0.0, 
                    max_value=10.0, 
                    step=0.01,
                    key='dop_altitude',
                    help="Enter altitude above sea level in kilometers"
                )
            
            st.markdown("")
            
            elevation_mask = st.slider(
                "Elevation Mask (deg)",
                min_value=0.0,
                max_value=30.0,
                step=1.0,
                key='dop_elevation_mask',
                help="Minimum elevation angle to consider a satellite visible"
            )
            
            st.markdown("")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                calculate_button = st.button("🔍 Calculate DOP", type="primary", use_container_width=True)
            
            auto_calculate_dop = st.session_state.pop('dop_auto_calculate', False)

            if calculate_button or auto_calculate_dop:
                with st.spinner("Calculating DOP values..."):
                    # Build combined parsed data if requested
                    dop_parsed = dict(parsed_data)  # Start with current constellation
                    dop_sats_lookup = dict(active_sats)  # For PRN lookups
                    
                    if use_combined:
                        for other_constellation in selected_combined_constellations:
                            other_tle_data = st.session_state[f'tle_data_{other_constellation}']
                            other_cfg = CONSTELLATION_CONFIG[other_constellation]
                            other_sats = other_cfg['sats']

                            for svn, tle in other_tle_data.items():
                                params = parse_tle_record(svn, tle)
                                if params:
                                    try:
                                        dop_parsed[svn] = propagate_tle_with_sgp4(params, analysis_epoch, svn)
                                    except Exception as exc:
                                        st.warning(f"Skipping {svn} in combined DOP: SGP4 propagation failed ({exc})")
                                        continue
                                    if svn in other_sats:
                                        dop_sats_lookup[svn] = other_sats[svn]
                    
                    dop_results = calculate_dop(
                        dop_parsed, 
                        latitude, 
                        longitude, 
                        altitude, 
                        elevation_mask,
                        active_sats=dop_sats_lookup
                    )
                
                st.session_state['dop_results'] = dop_results
                st.session_state['dop_location'] = (latitude, longitude, altitude)
                st.session_state['dop_sats_lookup'] = dop_sats_lookup
                st.session_state['dop_mode'] = dop_mode_label
                st.session_state['dop_result_elevation_mask'] = elevation_mask

            st.markdown("### Earth Map")
            map_results = None
            stored_location = st.session_state.get('dop_location')
            stored_mask = st.session_state.get('dop_result_elevation_mask')
            if 'dop_results' in st.session_state and stored_location is not None and stored_mask is not None:
                result_lat, result_lon, result_alt = stored_location
                if (
                    abs(result_lat - latitude) < 1e-6
                    and abs(result_lon - longitude) < 1e-6
                    and abs(result_alt - altitude) < 1e-6
                    and abs(stored_mask - elevation_mask) < 1e-6
                    and st.session_state.get('dop_mode') == dop_mode_label
                ):
                    map_results = st.session_state['dop_results']

            offline_map = create_dop_offline_map(
                latitude, longitude, map_results,
                st.session_state.get('dop_sats_lookup', active_sats),
            )
            map_response = st.plotly_chart(
                offline_map, key="dop_offline_map", use_container_width=True,
                on_select="rerun", selection_mode="points",
            )
            st.caption("Click the map to select an observer on its 2° grid; enter exact coordinates above. Coastlines are bundled locally.")
            selected_points = map_response.get("selection", {}).get("points", [])
            if selected_points and selected_points[-1].get("customdata") is not None:
                clicked_lat, clicked_lon = map(float, selected_points[-1]["customdata"])
                click_signature = (round(clicked_lat, 6), round(clicked_lon, 6), elevation_mask, dop_mode_label)
                if st.session_state.get('dop_last_map_click') != click_signature:
                    st.session_state['dop_last_map_click'] = click_signature
                    st.session_state['dop_pending_map_location'] = {'lat': clicked_lat, 'lon': clicked_lon}
                    rerun_streamlit()
            
            # Display results if available
            if 'dop_results' in st.session_state:
                dop_results = st.session_state['dop_results']
                loc_lat, loc_lon, loc_alt = st.session_state['dop_location']
                dop_sats_lookup = st.session_state.get('dop_sats_lookup', active_sats)
                dop_mode = st.session_state.get('dop_mode', constellation_choice)
                
                st.markdown("---")
                
                st.markdown(f"""
                <div style="padding: 20px; background: linear-gradient(135deg, rgba(76,175,80,0.2), rgba(56,142,60,0.1)); border-radius: 12px; margin: 20px 0;">
                    <h3 style="margin: 0 0 10px 0; color: #4CAF50;">📊 DOP Results ({dop_mode})</h3>
                    <p style="margin: 0; color: #ddd; font-size: 1.1em;">📍 Location: {loc_lat:.4f}°, {loc_lon:.4f}° (Altitude: {loc_alt:.3f} km)</p>
                </div>
                """, unsafe_allow_html=True)
                
                if 'error' in dop_results:
                    st.error(f"⚠️ {dop_results['error']}")
                    st.warning(f"🛰️ Visible satellites: {dop_results['visible_satellites']} (need at least 4)")
                else:
                    st.markdown("")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    def get_dop_quality(value):
                        if value < 1:
                            return "Ideal", "⭐", "#00E676", "Highest possible confidence level for applications demanding the highest possible precision at all times."
                        elif value <= 2:
                            return "Excellent", "🟢", "#4CAF50", "Positional measurements are considered accurate enough to meet all but the most sensitive applications."
                        elif value <= 5:
                            return "Good", "🟡", "#FFEB3B", "Minimum appropriate level for making accurate decisions. Suitable for reliable in-route navigation suggestions."
                        elif value <= 10:
                            return "Moderate", "🟠", "#FF9800", "Measurements could be used for calculations, but fix quality could still be improved. A more open view of the sky is recommended."
                        elif value <= 20:
                            return "Fair", "🔴", "#F44336", "Low confidence level. Positional measurements should be discarded or used only to indicate a very rough estimate of the current location."
                        else:
                            return "Poor", "⚫", "#9E9E9E", "At this level, measurements should be discarded."
                    
                    dop_cols = [
                        (col1, 'GDOP', dop_results['GDOP'], '#64B5F6'),
                        (col2, 'PDOP', dop_results['PDOP'], '#81C784'),
                        (col3, 'HDOP', dop_results['HDOP'], '#FFB74D'),
                        (col4, 'VDOP', dop_results['VDOP'], '#F06292'),
                        (col5, 'TDOP', dop_results['TDOP'], '#BA68C8')
                    ]
                    
                    for col, label, value, color in dop_cols:
                        with col:
                            quality, emoji, qual_color, description = get_dop_quality(value)
                            st.markdown(f"""
                            <div style="padding: 20px; background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)); 
                                        border-radius: 12px; text-align: center; border-left: 4px solid {color};">
                                <p style="margin: 0; color: {color}; font-size: 0.9em; font-weight: bold;">{label}</p>
                                <p style="margin: 10px 0; color: white; font-size: 2em; font-weight: bold;">{value:.2f}</p>
                                <p style="margin: 0; color: {qual_color}; font-size: 0.85em;">{emoji} {quality}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("")
                    
                    best_dop = min(dop_results['GDOP'], dop_results['PDOP'], dop_results['HDOP'], dop_results['VDOP'], dop_results['TDOP'])
                    overall_quality, overall_emoji, overall_color, overall_desc = get_dop_quality(best_dop)
                    
                    st.markdown(f"""
                    <div style="padding: 20px; background: linear-gradient(135deg, rgba({int(overall_color[1:3], 16)},{int(overall_color[3:5], 16)},{int(overall_color[5:7], 16)},0.2), rgba({int(overall_color[1:3], 16)},{int(overall_color[3:5], 16)},{int(overall_color[5:7], 16)},0.1)); border-radius: 12px; margin: 10px 0;">
                        <h4 style="margin: 0 0 10px 0; color: {overall_color};">{overall_emoji} Overall Quality Assessment: {overall_quality}</h4>
                        <p style="margin: 0; color: #ddd; font-size: 0.95em; line-height: 1.6;">{overall_desc}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    st.markdown(f"""
                    <div style="padding: 15px; background: linear-gradient(90deg, rgba(156,39,176,0.2), rgba(123,31,162,0.1)); border-radius: 10px; margin: 20px 0 15px 0;">
                        <h3 style="margin: 0; color: white;">🛰️ Visible Satellites ({dop_results['visible_satellites']})</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if dop_results['satellite_details']:
                        col_left, col_right = st.columns([1, 1])
                        
                        with col_left:
                            sat_df = pd.DataFrame([
                                {
                                    'SVN': sat['svn'],
                                    'PRN': dop_sats_lookup[sat['svn']]['prn'] if sat['svn'] in dop_sats_lookup else 'N/A',
                                    'Azimuth (°)': f"{sat['azimuth']:.1f}",
                                    'Elevation (°)': f"{sat['elevation']:.1f}",
                                    'Range (km)': f"{sat['range']:.0f}",
                                }
                                for sat in dop_results['satellite_details']
                            ])
                            
                            st.markdown("**📋 Satellite List**")
                            st.dataframe(sat_df, hide_index=True, use_container_width=True, height=400)
                        
                        with col_right:
                            st.markdown("**🌌 Sky Plot**")
                            
                            fig = go.Figure()
                            
                            for sat in dop_results['satellite_details']:
                                radius = 90 - sat['elevation']
                                
                                if sat['elevation'] >= 60:
                                    color = '#4CAF50'
                                elif sat['elevation'] >= 30:
                                    color = '#FFEB3B'
                                else:
                                    color = '#FF9800'
                                
                                fig.add_trace(go.Scatterpolar(
                                    r=[radius],
                                    theta=[sat['azimuth']],
                                    mode='markers+text',
                                    name=sat['svn'],
                                    text=[dop_sats_lookup[sat['svn']]['prn'] if sat['svn'] in dop_sats_lookup else sat['svn']],
                                    textposition="top center",
                                    textfont=dict(size=10, color='white'),
                                    marker=dict(size=12, symbol='diamond', color=color, line=dict(width=1, color='white')),
                                    hovertemplate=f"<b>{sat['svn']}</b><br>" +
                                                f"PRN: {dop_sats_lookup[sat['svn']]['prn'] if sat['svn'] in dop_sats_lookup else 'N/A'}<br>" +
                                                f"Azimuth: {sat['azimuth']:.1f}°<br>" +
                                                f"Elevation: {sat['elevation']:.1f}°<br>" +
                                                f"Range: {sat['range']:.0f} km<extra></extra>"
                                ))
                            
                            fig.update_layout(
                                polar=dict(
                                    radialaxis=dict(
                                        visible=True,
                                        range=[0, 90],
                                        tickvals=[0, 30, 60, 90],
                                        ticktext=['90°', '60°', '30°', '0°'],
                                        tickmode='array',
                                        gridcolor='rgba(255,255,255,0.2)'
                                    ),
                                    angularaxis=dict(
                                        tickmode='array',
                                        tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                                        ticktext=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
                                        direction='clockwise',
                                        rotation=90,
                                        gridcolor='rgba(255,255,255,0.2)'
                                    ),
                                    bgcolor='rgba(0,0,0,0.3)'
                                ),
                                showlegend=False,
                                title="Observer's View<br><sub>Green: El>60° | Yellow: El>30° | Orange: El<30°</sub>",
                                height=450,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white')
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Summary statistics
                        st.markdown("")
                        st.markdown("**📈 Coverage Statistics**")
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        
                        with stat_col1:
                            avg_elevation = np.mean([s['elevation'] for s in dop_results['satellite_details']])
                            st.metric("Avg Elevation", f"{avg_elevation:.1f}°")
                        
                        with stat_col2:
                            max_elevation = max([s['elevation'] for s in dop_results['satellite_details']])
                            st.metric("Max Elevation", f"{max_elevation:.1f}°")
                        
                        with stat_col3:
                            avg_range = np.mean([s['range'] for s in dop_results['satellite_details']])
                            st.metric("Avg Range", f"{avg_range:.0f} km")
                        
                        with stat_col4:
                            high_elev = len([s for s in dop_results['satellite_details'] if s['elevation'] >= 30])
                            st.metric("High Elevation", f"{high_elev}/{len(dop_results['satellite_details'])}")
                    

if __name__ == "__main__":
    main()

