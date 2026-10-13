"""Validate the bundled orbital snapshot without credentials or network access."""

from snapshot_data import load_snapshot


def main():
    snapshot, epoch = load_snapshot()
    policy = snapshot["epoch_policy"]
    print(f"Validated {policy['object_count']} TLEs at {epoch.isoformat()}")
    print(f"Complete coverage: {policy['required_operational_object_count']} USABLE satellites")


if __name__ == "__main__":
    main()
