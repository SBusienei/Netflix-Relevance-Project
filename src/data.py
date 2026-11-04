"""Keep retrieval cleaning explicit and auditable without changing the source CSV."""

import csv
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parents[1] / "netflix_titles.csv"
FIELDS = (
    "show_id", "type", "title", "director", "cast", "country", "date_added",
    "release_year", "rating", "duration", "listed_in", "description",
)
FilmRecord = dict[str, str]


@dataclass(frozen=True)
class CleaningReport:
    input_rows: int
    output_rows: int
    duplicates_removed: int
    missing_before: dict[str, int]
    missing_after: dict[str, int]
    normalized_cells: dict[str, int]
    duration_repairs: tuple[str, ...]


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def missing_counts(records: list[FilmRecord]) -> dict[str, int]:
    return {
        field: sum(not record[field].strip() for record in records)
        for field in FIELDS
    }


def read_records(path: str | Path) -> list[FilmRecord]:
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        if reader.fieldnames != list(FIELDS):
            raise ValueError(f"Expected CSV columns in this order: {', '.join(FIELDS)}")
        records: list[FilmRecord] = []
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV record at record number {line}")
            records.append({field: row[field] for field in FIELDS})
    if not records:
        raise ValueError("The film CSV contains no records")
    return records


def clean_records(records: list[FilmRecord]) -> tuple[list[FilmRecord], CleaningReport]:
    if not records:
        raise ValueError("No film records to clean")
    cleaned: list[FilmRecord] = []
    seen_rows: set[tuple[str, ...]] = set()
    seen_ids: set[str] = set()
    normalized: Counter[str] = Counter()
    repairs: list[str] = []
    for source in records:
        if set(source) != set(FIELDS):
            raise ValueError("Every record must contain exactly the expected CSV fields")
        record = {field: normalize_text(source[field]) for field in FIELDS}
        normalized.update(field for field in FIELDS if source[field] != record[field])
        if not record["show_id"]:
            raise ValueError("A film record is missing show_id")
        if not record["duration"] and re.fullmatch(r"\d+ min", record["rating"]):
            record["duration"], record["rating"] = record["rating"], ""
            repairs.append(record["show_id"])
        signature = tuple(record[field] for field in FIELDS)
        if signature in seen_rows:
            continue
        if record["show_id"] in seen_ids:
            raise ValueError(f"Conflicting duplicate show_id: {record['show_id']}")
        seen_rows.add(signature)
        seen_ids.add(record["show_id"])
        cleaned.append(record)
    report = CleaningReport(
        input_rows=len(records),
        output_rows=len(cleaned),
        duplicates_removed=len(records) - len(cleaned),
        missing_before=missing_counts(records),
        missing_after=missing_counts(cleaned),
        normalized_cells={field: normalized[field] for field in FIELDS},
        duration_repairs=tuple(repairs),
    )
    return cleaned, report


def load_records(path: str | Path = DATA_PATH) -> tuple[list[FilmRecord], CleaningReport]:
    return clean_records(read_records(path))
