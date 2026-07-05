"""
OPTIONAL: skeleton scraper to regenerate data/catalog.json directly from
shl.com's Individual Test Solutions catalog, in case the assignment-provided
scrape goes stale. Not required to run the project — the assignment gives you a
catalog dump already; just save it to data/catalog.json (see README).

This is intentionally a skeleton, not a finished scraper: SHL's markup can change,
and re-scraping 500+ product pages is slow and easy to get rate-limited on. Fill
in the CSS selectors for the fields you need, run it from an environment with
normal internet access (this repo's sandbox network is restricted to package
registries only, not shl.com), and point CATALOG_PATH at the output.

Usage (once selectors are filled in):
    pip install requests beautifulsoup4
    python scripts/scrape_catalog.py --out data/catalog.json
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List, Optional

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    requests = None
    BeautifulSoup = None

CATALOG_INDEX_URL = "https://www.shl.com/solutions/products/product-catalog/"


def fetch(url: str, session, retries: int = 3) -> Optional[str]:
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            return resp.text
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt+1}/{retries} for {url}: {e}")
            time.sleep(1.5 * (attempt + 1))
    return None


def list_product_urls(session) -> List[str]:
    """TODO: paginate the Individual Test Solutions catalog listing and collect
    every /products/product-catalog/view/<slug>/ URL. The listing is paginated
    with a `start=` or `?page=` query param on shl.com — inspect the live site
    and fill this in."""
    raise NotImplementedError(
        "Fill in pagination + link extraction for the current shl.com markup, "
        "or just use the assignment-provided catalog dump instead."
    )


def parse_product_page(html: str, url: str) -> Dict:
    """TODO: extract entity_id, name, description, job_levels, languages,
    duration, adaptive, remote, keys (test-type category chips) from a single
    product detail page. Inspect the live DOM and fill in selectors."""
    raise NotImplementedError("Fill in field extraction for the current shl.com markup.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/catalog.json")
    args = parser.parse_args()

    if requests is None:
        raise SystemExit("pip install requests beautifulsoup4 first")

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; SHL-catalog-refresh/1.0)"})

    urls = list_product_urls(session)
    records = []
    for i, url in enumerate(urls, start=1):
        print(f"[{i}/{len(urls)}] {url}")
        html = fetch(url, session)
        if html is None:
            records.append({"link": url, "status": "fetch_failed"})
            continue
        try:
            records.append(parse_product_page(html, url))
        except Exception as e:  # noqa: BLE001
            records.append({"link": url, "status": f"parse_failed: {e}"})
        time.sleep(0.5)  # be polite

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"Wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
