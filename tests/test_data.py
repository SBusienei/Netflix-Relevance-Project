"""Protect corpus identity and make every automatic cleaning rule testable."""

import csv
from pathlib import Path

import pytest

from src.data import FIELDS, FilmRecord, clean_records, load_records, normalize_text


def record(show_id: str = "s1", **values: str) -> FilmRecord:
    return dict.fromkeys(FIELDS, "") | {"show_id": show_id} | values


def write_csv(path: Path, records: list[FilmRecord]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def test_normalization_preserves_case_accents_and_punctuation() -> None:
    assert normalize_text("  Cafe\u0301\tO'Brien,\nJr.  ") == "Caf\u00e9 O'Brien, Jr."


def test_cleaning_preserves_source_and_does_not_invent_missing_values() -> None:
    source = record(title="  Cafe\u0301\tStory ", description="Line\none")
    cleaned, report = clean_records([source])
    assert source["title"] == "  Cafe\u0301\tStory "
    assert cleaned[0]["title"] == "Caf\u00e9 Story"
    assert cleaned[0]["cast"] == ""
    assert report.normalized_cells["title"] == 1
    assert report.normalized_cells["description"] == 1
    assert report.missing_before["cast"] == report.missing_after["cast"] == 1


def test_only_exact_cleaned_duplicates_are_removed() -> None:
    first = record(title="Story")
    cleaned, report = clean_records([first, record(title=" Story "), record("s2", title="Story")])
    assert [row["show_id"] for row in cleaned] == ["s1", "s2"]
    assert report.duplicates_removed == 1


def test_conflicting_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="Conflicting duplicate show_id: s1"):
        clean_records([record(title="One"), record(title="Two")])


def test_missing_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing show_id"):
        clean_records([record("  ")])


def test_unexpected_record_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="expected CSV fields"):
        clean_records([{"show_id": "s1"}])


def test_duration_repair_is_narrow_and_reported() -> None:
    cleaned, report = clean_records([
        record(rating="74 min"),
        record("s2", rating="84 min", duration="90 min"),
        record("s3", rating="TV-MA"),
    ])
    assert (cleaned[0]["rating"], cleaned[0]["duration"]) == ("", "74 min")
    assert (cleaned[1]["rating"], cleaned[1]["duration"]) == ("84 min", "90 min")
    assert cleaned[2]["rating"] == "TV-MA"
    assert report.duration_repairs == ("s1",)
    assert report.missing_before["duration"] == 2
    assert report.missing_after["duration"] == 1
    assert report.missing_after["rating"] == 1


def test_utf8_bom_and_csv_quoted_fields(tmp_path: Path) -> None:
    path = tmp_path / "films.csv"
    write_csv(path, [record(title="One, Two", cast="A, B", description="First\nsecond")])
    cleaned, _ = load_records(path)
    assert cleaned[0]["title"] == "One, Two"
    assert cleaned[0]["cast"] == "A, B"
    assert cleaned[0]["description"] == "First second"


def test_literal_na_is_not_treated_as_missing(tmp_path: Path) -> None:
    path = tmp_path / "films.csv"
    write_csv(path, [record(title="NA", country="NA")])
    cleaned, report = load_records(path)
    assert cleaned[0]["title"] == cleaned[0]["country"] == "NA"
    assert report.missing_after["country"] == 0


@pytest.mark.parametrize("text", [
    "",
    "show_id,title\ns1,Story\n",
    ",".join(FIELDS) + "\ns1,Too few\n",
    ",".join(FIELDS) + "\n" + ",".join(["value"] * (len(FIELDS) + 1)),
    ",".join(FIELDS) + "\n",
])
def test_malformed_or_empty_csv_is_rejected(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_records(path)


def test_unterminated_quoted_cell_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    text = ",".join(FIELDS) + "\n" + ",".join(["value"] * (len(FIELDS) - 1)) + ',"unfinished'
    path.write_text(text, encoding="utf-8")
    with pytest.raises(csv.Error):
        load_records(path)


def test_repository_corpus_audit() -> None:
    records, report = load_records()
    assert report.input_rows == report.output_rows == 8807
    assert report.duplicates_removed == 0
    assert len({row["show_id"] for row in records}) == 8807
    assert sum(row["type"] == "Movie" for row in records) == 6131
    assert sum(row["type"] == "TV Show" for row in records) == 2676
    assert report.duration_repairs == ("s5542", "s5795", "s5814")
    assert report.missing_before == {
        "show_id": 0, "type": 0, "title": 0, "director": 2634, "cast": 825,
        "country": 831, "date_added": 10, "release_year": 0, "rating": 4,
        "duration": 3, "listed_in": 0, "description": 0,
    }
    assert report.missing_after == report.missing_before | {"rating": 7, "duration": 0}
