"""Strict, network-free validation for the bundled GNSS snapshot."""

import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path


SNAPSHOT_PATH = Path(__file__).with_name("static_snapshot.json")
CONSTELLATIONS = ("Galileo", "BeiDou", "GPS", "GLONASS")


def parse_utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include its UTC offset.")
    return parsed.astimezone(UTC)


def tle_epoch(line):
    year = int(line[18:20])
    year += 2000 if year < 57 else 1900
    day = float(line[20:32])
    if not math.isfinite(day) or not 1 <= day < 367:
        raise ValueError("Invalid TLE epoch day.")
    return datetime(year, 1, 1, tzinfo=UTC) + timedelta(days=day - 1)


def valid_checksum(line):
    if not isinstance(line, str) or len(line) != 69 or not line[-1].isdigit():
        return False
    total = sum(int(char) if char.isdigit() else int(char == "-") for char in line[:-1])
    return total % 10 == int(line[-1])


def validate_snapshot(snapshot):
    """Reject corruption, missing operational coverage, and mixed-epoch data."""
    if snapshot.get("schema_version") != 1:
        raise ValueError("Unsupported snapshot schema.")
    canonical = {key: value for key, value in snapshot.items() if key != "sha256"}
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != snapshot.get("sha256"):
        raise ValueError("Snapshot integrity check failed.")
    analysis_epoch = parse_utc(snapshot["analysis_epoch"])
    limit = float(snapshot["epoch_policy"]["max_offset_hours"])
    if not math.isfinite(limit) or not 0 < limit <= 72:
        raise ValueError("Snapshot must keep every element within 72 hours of the common epoch.")
    if set(snapshot["constellations"]) != set(CONSTELLATIONS):
        raise ValueError("Snapshot must contain all four supported constellations.")

    seen = set()
    operational_count = 0
    for name, dataset in list(snapshot["constellations"].items()) + [("GPS legacy", snapshot["gps_legacy"])]:
        roster, tles = dataset["roster"], dataset["tles"]
        if not roster or not isinstance(roster, dict) or not isinstance(tles, dict):
            raise ValueError(f"Missing {name} roster or elements.")
        if set(tles) - set(roster):
            raise ValueError(f"{name} contains an element outside its frozen roster.")
        for key, entry in roster.items():
            if entry.get("status") == "USABLE":
                operational_count += 1
                if key not in tles:
                    raise ValueError(f"Missing operational element for {name} / {key}.")
            if key not in tles:
                continue
            record = tles[key]
            norad = int(entry["norad"])
            if norad in seen:
                raise ValueError(f"Repeated NORAD catalogue ID: {norad}.")
            seen.add(norad)
            for number in (1, 2):
                line = record[f"line{number}"]
                if not valid_checksum(line) or not line.startswith(f"{number} ") or int(line[2:7]) != norad:
                    raise ValueError(f"Invalid TLE checksum or identity for {key}.")
            epoch = tle_epoch(record["line1"])
            if abs((epoch - parse_utc(record["epoch"])).total_seconds()) > 0.001:
                raise ValueError(f"Stored epoch disagrees with the raw TLE for {key}.")
            if abs((epoch - analysis_epoch).total_seconds()) > limit * 3600:
                raise ValueError(f"TLE outside the common-epoch window: {key}.")
        for key, date in dataset.get("launch_dates", {}).items():
            if date:
                launch = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=UTC)
                if launch > analysis_epoch:
                    raise ValueError(f"Launch date after snapshot epoch: {key}.")
    policy = snapshot["epoch_policy"]
    if len(seen) != policy["object_count"] or operational_count != policy["required_operational_object_count"]:
        raise ValueError("Snapshot coverage counts disagree with its contents.")
    return analysis_epoch


def load_snapshot(path=SNAPSHOT_PATH):
    try:
        with Path(path).open(encoding="utf-8") as handle:
            snapshot = json.load(handle)
        epoch = validate_snapshot(snapshot)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        raise RuntimeError(f"Could not load the bundled static snapshot: {exc}") from exc
    return snapshot, epoch
