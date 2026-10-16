"""Physical geometry cases for the static dashboard's orbital diagnostics."""

import numpy as np
import pytest

import static_app as app


def circular_state(raan=0.0, inclination=55.0, phase=0.0, radius=26560.0,
                   mean_motion=2.0056):
    node, inc, u = np.radians([raan, inclination, phase])
    p = np.array([np.cos(node), np.sin(node), 0.0])
    q = np.array([-np.sin(node) * np.cos(inc),
                  np.cos(node) * np.cos(inc), np.sin(inc)])
    return {
        'position_eci_km': radius * (p * np.cos(u) + q * np.sin(u)),
        'velocity_eci_km_s': 4.0 * (-p * np.sin(u) + q * np.cos(u)),
        'raan': raan, 'inclination': inclination, 'argument_latitude': phase,
        'mean_motion': mean_motion, 'mean_anomaly': phase,
    }


def test_tilted_plane_crossing_has_zero_distance_but_nonzero_plane_angle():
    reference = app.orbital_plane_normal(0, 55)
    state = circular_state(inclination=58, phase=0)
    result = app.calculate_drift(state, reference_plane_normal=reference, plane_limit_deg=2)
    assert result['cross_track_km'] == pytest.approx(0, abs=1e-9)
    assert result['plane_angle_deg'] == pytest.approx(3)
    assert not result['across_track_ok']


@pytest.mark.parametrize('phase, sign', [(90, 1), (270, -1)])
def test_signed_cross_track_uses_actual_radius_and_orbital_phase(phase, sign):
    state = circular_state(inclination=58, phase=phase, radius=30000)
    result = app.calculate_drift(state, orbital_radius=26560,
                                 reference_plane_normal=app.orbital_plane_normal(0, 55))
    assert result['cross_track_km'] == pytest.approx(sign * 30000 * np.sin(np.radians(3)))
    assert result['plane_angle_deg'] == pytest.approx(3)


def test_raan_separation_respects_inclination_and_wraparound():
    # At low inclination, even a 20-degree node difference is a small plane tilt.
    state = circular_state(raan=10, inclination=5, phase=45)
    reference = app.orbital_plane_normal(350, 5)
    result = app.calculate_drift(state, reference_plane_normal=reference, plane_limit_deg=2)
    expected_angle = 2 * np.degrees(np.arcsin(np.sin(np.radians(5)) * np.sin(np.radians(10))))
    assert result['delta_raan'] == pytest.approx(20)
    assert result['plane_angle_deg'] == pytest.approx(expected_angle)
    assert result['across_track_ok']


def test_projection_and_plane_angle_are_invariant_under_shared_frame_rotation():
    state = circular_state(raan=100, inclination=58, phase=110)
    reference = app.orbital_plane_normal(98, 55)
    expected = app.calculate_plane_deviation(state, reference)
    # A rigid, orthogonal rotation applied to both states and reference.
    angle = np.radians(43)
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    rotated = dict(state)
    for key in ('position_eci_km', 'velocity_eci_km_s'):
        rotated[key] = rotation @ state[key]
    actual = app.calculate_plane_deviation(rotated, rotation @ reference)
    assert actual == pytest.approx(expected, abs=1e-9)


def test_peer_fit_absorbs_common_tilt_while_design_plane_reveals_it():
    states = {str(u): circular_state(inclination=58, phase=u) for u in (0, 90, 180)}
    states['excluded'] = circular_state(inclination=70)
    assignments = {svn: 1 for svn in states}
    peer = app.fit_peer_plane_normals(states, assignments, ['0', '90', '180'])[1]
    result = app.calculate_drift(states['90'], reference_plane_normal=peer,
                                 nominal_plane_normal=app.orbital_plane_normal(0, 55))
    assert result['plane_angle_deg'] == pytest.approx(0, abs=1e-10)
    assert result['cross_track_km'] == pytest.approx(0, abs=1e-9)
    assert result['nominal_plane_angle_deg'] == pytest.approx(3)
    assert result['nominal_cross_track_km'] > 1000


def test_missing_reference_is_unavailable_and_degenerate_state_is_rejected():
    result = app.calculate_drift(circular_state())
    assert np.isnan(result['cross_track_km'])
    assert np.isnan(result['plane_angle_deg'])
    assert np.isnan(result['nominal_cross_track_km'])
    assert result['across_track_ok'] is None
    state = circular_state()
    state['velocity_eci_km_s'] = np.zeros(3)
    with pytest.raises(ValueError, match='nonzero'):
        app.calculate_plane_deviation(state, [0, 0, 1])


@pytest.mark.parametrize('delta_n', [-0.001, 0.0, 0.001])
def test_mean_motion_rate_units_sign_and_configurable_limit(delta_n):
    state = circular_state(mean_motion=2.0 + delta_n)
    result = app.calculate_drift(state, nominal_mean_motion=2.0,
                                 orbital_radius=30000, drift_limit_deg_day=0.2)
    assert result['drift_angle_per_day'] == pytest.approx(360 * delta_n, abs=1e-12)
    assert result['in_plane_drift_km_day'] == pytest.approx(2 * np.pi * 30000 * delta_n, abs=1e-9)
    assert result['in_plane_ok'] == (abs(delta_n) < 0.0005)


def test_configured_gps_design_normal_is_in_the_propagated_state_frame():
    normal = app.configured_nominal_plane_normals('GPS', {1: 288.85}, 55)[1]
    # Convert back to the specification axes and recover its prescribed normal.
    assert app.ICRS_to_J2000 @ normal == pytest.approx(app.orbital_plane_normal(288.85, 55), abs=1e-14)
