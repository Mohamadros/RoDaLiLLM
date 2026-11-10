"""Download the three public HPI CD Dataset 1 TSV files with hash checks."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

BASE = "https://hpi.de/oldsite/fileadmin/user_upload/fachgebiete/naumann/projekte/repeatability/freeDB"
FILES = {
    "cddb.tsv": (
        f"{BASE}/cddb.tsv",
        "3220f733bfd3039dc14eafca7ccede154449b1ea726debbae8b5331365f374cd",
    ),
    "cddb_DPL.tsv": (
        f"{BASE}/cddb_DPL.tsv",
        "2dc82b3c1e1fce429c1aa0a459d14915856bb87ceeaf43e3a76564813da07a53",
    ),
    # HPI labels this mdedup link as the updated, simplified NDPL approach.
    "cddb_NDPL.tsv": (
        f"{BASE}/mdedup/cddb_NDPL.tsv",
        "79108daf79c35a84df3d771cec5f373269a1e0dd24d89c03a4fd6869e4a81d29",
    ),
}


def main() -> None:
    destination = Path("data/raw")
    destination.mkdir(parents=True, exist_ok=True)
    for name, (url, expected_hash) in FILES.items():
        target = destination / name
        request = urllib.request.Request(
            url, headers={"User-Agent": "RoDaLiLLM-CD-Benchmark/0.1"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Checksum mismatch for {name}: expected {expected_hash}, got {actual_hash}"
            )
        target.write_bytes(content)
        print(f"Downloaded {name} ({len(content):,} bytes)")


if __name__ == "__main__":
    main()
