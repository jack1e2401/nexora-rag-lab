"""Ingest one fixed-token benchmark profile into pgvector."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from app.config import get_settings
from app.ingestion.pipeline import ingest_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/benchmark/manifest.json"))
    parser.add_argument("--profile", required=True)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--reset-corpus", action="store_true")
    args = parser.parse_args()

    report = ingest_manifest(
        args.manifest, args.profile, args.chunk_size, args.overlap,
        get_settings(), args.reset_corpus,
    )
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
