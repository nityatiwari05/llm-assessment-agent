"""
Sanity-check your data/catalog.json before deploying. Run this after saving the
assignment's catalog JSON to data/catalog.json.

Usage:
    python scripts/validate_catalog.py
    python scripts/validate_catalog.py path/to/other_catalog.json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.catalog import load_catalog  # noqa: E402


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    try:
        items = load_catalog(path)
    except Exception as e:  # noqa: BLE001
        print(f"FAILED to load catalog: {e}")
        sys.exit(1)

    print(f"Loaded {len(items)} catalog items.")

    no_desc = [i for i in items if not i.description]
    no_type = [i for i in items if not i.test_type]
    bad_url = [i for i in items if not i.url.startswith("http")]

    print(f"  items with empty description: {len(no_desc)}")
    print(f"  items with no derived test_type: {len(no_type)}")
    print(f"  items with suspicious URL: {len(bad_url)}")

    for name in [
        "Occupational Personality Questionnaire OPQ32r",
        "Core Java (Advanced Level) (New)",
        "SHL Verify Interactive G+",
        "Graduate Scenarios",
    ]:
        found = any(i.name == name for i in items)
        print(f"  contains '{name}': {'yes' if found else 'MISSING'}")

    if bad_url:
        print("\nSample bad URLs:")
        for i in bad_url[:5]:
            print(f"    {i.entity_id}: {i.url!r}")

    print("\nOK — catalog looks loadable. Run `pytest` and the eval scripts next.")


if __name__ == "__main__":
    main()
