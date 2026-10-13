"""Exercise the actual bundled snapshot and dashboard with network calls blocked."""

import contextlib
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import io
import json
from pathlib import Path
import socket
from unittest.mock import patch

import numpy as np
import pytest
import requests
from streamlit.testing.v1 import AppTest

import static_app as app
import snapshot_data


APP_PATH = Path(app.__file__).resolve()


def sign(snapshot):
    payload = {key: value for key, value in snapshot.items() if key != "sha256"}
    snapshot["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return snapshot


def test_real_snapshot_is_complete_and_propagates_offline():
    snapshot, epoch = snapshot_data.load_snapshot()
    count = 0
    total = 0
    with patch.object(socket.socket, "connect", side_effect=AssertionError("Unexpected network connection")):
        datasets = list(snapshot["constellations"].items()) + [("GPS legacy", snapshot["gps_legacy"])]
        for name, dataset in datasets:
            raw = {svn: app.parse_tle_record(svn, tle) for svn, tle in dataset["tles"].items()}
            states, errors = app.propagate_all_tles_with_sgp4(raw, epoch)
            assert not errors, (name, errors)
            assert set(states) == set(dataset["tles"])
            for svn, state in states.items():
                assert state["analysis_epoch"] == epoch
                assert np.all(np.isfinite(state["position_ecef_km"]))
                count += dataset["roster"][svn].get("status") == "USABLE"
                total += 1
    assert count == 111
    assert total == 125
    assert len(snapshot["gps_legacy"]["tles"]) == 7


def test_validation_rejects_missing_operational_data_and_epoch_changes():
    snapshot, _ = snapshot_data.load_snapshot()
    missing = deepcopy(snapshot)
    del missing["constellations"]["GPS"]["tles"]["GPS-PRN01"]
    with pytest.raises(ValueError, match="Missing operational element"):
        snapshot_data.validate_snapshot(sign(missing))
    shifted = deepcopy(snapshot)
    shifted["analysis_epoch"] = "2026-10-10T00:00:00Z"
    with pytest.raises(ValueError, match="outside the common-epoch window"):
        snapshot_data.validate_snapshot(sign(shifted))
    tampered = deepcopy(snapshot)
    tampered["analysis_epoch"] = "2026-10-10T00:00:00Z"
    with pytest.raises(ValueError, match="integrity"):
        snapshot_data.validate_snapshot(tampered)


def test_static_age_is_independent_of_timezone_and_current_date():
    _, epoch = snapshot_data.load_snapshot()
    app.STATIC_ANALYSIS_EPOCH = epoch
    naive = datetime(2015, 3, 25)
    aware = naive.replace(tzinfo=UTC)
    assert app.calculate_satellite_age_score(naive) == app.calculate_satellite_age_score(aware)


def test_every_constellation_and_combined_dop_without_parent_files_or_network(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with (
        patch.object(socket.socket, "connect", side_effect=AssertionError("Unexpected network connection")),
        patch.object(requests.Session, "request", side_effect=AssertionError("Unexpected provider request")),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        dashboard = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
        assert not dashboard.exception, [item.message for item in dashboard.exception]
        assert not dashboard.error, [item.value for item in dashboard.error]
        for name in snapshot_data.CONSTELLATIONS:
            dashboard.selectbox(key="constellation_selector").set_value(name).run()
            assert not dashboard.exception, (name, [item.message for item in dashboard.exception])
            assert not dashboard.error, (name, [item.value for item in dashboard.error])
            assert len(dashboard.tabs) >= 4
            assert any("2026-07-14 12:00:00 UTC" in item.value for item in dashboard.caption)
            assert not dashboard.text_input
            assert not any("Fetch" in item.label or "Update" in item.label for item in dashboard.button)
            assert not any("not found" in item.value for item in dashboard.warning)
            dashboard.number_input(key="dop_latitude").set_value(28.6139)
            dashboard.number_input(key="dop_longitude").set_value(77.2090)
            [item for item in dashboard.button if "Calculate DOP" in item.label][0].click().run()
            assert not dashboard.exception
            result = dashboard.session_state["dop_results"]
            assert "error" not in result, (name, result)
            assert result["visible_satellites"] >= 4
            assert result["GDOP"] > 0

        # All other datasets are available at startup, without visiting them.
        dashboard.multiselect[0].set_value(["Galileo", "BeiDou", "GPS"]).run()
        [item for item in dashboard.button if "Calculate DOP" in item.label][0].click().run()
        assert not dashboard.exception
        combined = dashboard.session_state["dop_results"]
        assert "error" not in combined
        assert combined["visible_satellites"] >= 20
        snapshot, _ = snapshot_data.load_snapshot()
        usable = {svn for dataset in snapshot["constellations"].values() for svn, entry in dataset["roster"].items() if entry["status"] == "USABLE"}
        assert all(item["svn"] in usable for item in combined["satellite_details"])
        assert not any(item["svn"].startswith("GPS-LEGACY") for item in combined["satellite_details"])
