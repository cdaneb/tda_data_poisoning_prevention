"""Read-only structural audit of materials supplied by Monkam et al.

The script intentionally does not execute notebook code or modify source files.
It emits JSON to stdout, or to a path supplied with ``--output``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import struct
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET


NOTEBOOK_LANDMARK = re.compile(
    r"(?i)(DBSCAN|HDBSCAN|OPTICS|MeanShift|estimate_bandwidth|CubicalPersistence|"
    r"PersistenceEntropy|Amplitude|BettiCurve|Scaler|threshold|poison|attack|"
    r"train_test_split|RandomForest|classification_report|confusion_matrix)"
)
FILE_LITERAL = re.compile(
    r"(?i)(?:[rubf]{0,2})?['\"]([^'\"\r\n]+\.(?:csv|xlsx?|npy|npz|pkl|pickle|json|png|pdf))['\"]"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def binary_line_count(path: Path) -> int:
    count = 0
    last = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            count += chunk.count(b"\n")
            last = chunk[-1:] or last
    return count + (1 if path.stat().st_size and last != b"\n" else 0)


def csv_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        first_row = next(reader, [])
    line_count = binary_line_count(path)
    return {
        "columns": len(header),
        "rows_excluding_header": max(0, line_count - 1),
        "header_first_12": header[:12],
        "header_last_12": header[-12:],
        "first_row_first_12": first_row[:12],
        "first_row_last_12": first_row[-12:],
    }


def poisoned_payload_vectors(source_dir: Path) -> dict[str, list[int]]:
    vectors: dict[str, list[int]] = {}
    for path in sorted(source_dir.glob("final_payload_*.csv")):
        values: list[int] = []
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for line in stream:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                values.append(round(float(stripped) * 255))
        vectors[path.name] = values
    return vectors


def payload_dataset_profile(path: Path, vector_names: Iterable[str]) -> dict[str, Any]:
    source_indices = {
        int(match.group(1))
        for name in vector_names
        if (match := re.match(r"final_payload_(\d+)-", name))
    }
    labels: Counter[str] = Counter()
    fixed_block_labels: Counter[str] = Counter()
    outside_block_labels: Counter[str] = Counter()
    source_rows: dict[int, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        label_index = header.index("label")
        for row_index, row in enumerate(reader):
            label = row[label_index]
            labels[label] += 1
            if 50138 <= row_index < 71138:
                fixed_block_labels[label] += 1
            else:
                outside_block_labels[label] += 1
            if row_index in source_indices:
                source_rows[row_index] = {
                    "label": label,
                    "payload": [int(float(value)) for value in row[:1500]],
                    "metadata": dict(zip(header[1500:], row[1500:])),
                }

    return {
        "labels": dict(labels),
        "fixed_block_50138_71137_labels": dict(fixed_block_labels),
        "outside_fixed_block_labels": dict(outside_block_labels),
        "named_source_rows": source_rows,
    }


def compare_payload_csvs(left_path: Path, right_path: Path) -> dict[str, Any]:
    text_different_cells = 0
    semantic_different_cells = 0
    semantic_different_rows = 0
    row_count = 0
    first_semantic_difference: dict[str, Any] | None = None
    with left_path.open("r", encoding="utf-8-sig", newline="") as left_stream, right_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as right_stream:
        left_reader = csv.reader(left_stream)
        right_reader = csv.reader(right_stream)
        left_header = next(left_reader)
        right_header = next(right_reader)
        headers_equal = left_header == right_header
        text_columns = {
            index for index, name in enumerate(left_header) if name in {"protocol", "label"}
        }
        for row_index, pair in enumerate(zip(left_reader, right_reader)):
            left_row, right_row = pair
            row_count += 1
            row_semantic_difference = False
            for column_index, (left, right) in enumerate(zip(left_row, right_row)):
                if left == right:
                    continue
                text_different_cells += 1
                if column_index in text_columns:
                    equal = left == right
                else:
                    equal = float(left) == float(right)
                if not equal:
                    semantic_different_cells += 1
                    row_semantic_difference = True
                    if first_semantic_difference is None:
                        first_semantic_difference = {
                            "row": row_index,
                            "column": left_header[column_index],
                            "left": left,
                            "right": right,
                        }
            if len(left_row) != len(right_row):
                row_semantic_difference = True
                semantic_different_cells += abs(len(left_row) - len(right_row))
            if row_semantic_difference:
                semantic_different_rows += 1
        left_extra = sum(1 for _row in left_reader)
        right_extra = sum(1 for _row in right_reader)
    return {
        "headers_equal": headers_equal,
        "paired_data_rows": row_count,
        "left_extra_rows": left_extra,
        "right_extra_rows": right_extra,
        "text_different_cells": text_different_cells,
        "semantic_different_cells": semantic_different_cells,
        "semantic_different_rows": semantic_different_rows,
        "first_semantic_difference": first_semantic_difference,
    }


def compare_poisoned_vectors(
    vectors: dict[str, list[int]], profile: dict[str, Any]
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    source_rows = profile["named_source_rows"]
    for name, vector in vectors.items():
        match = re.match(r"final_payload_(\d+)-", name)
        source_index = int(match.group(1)) if match else None
        source = source_rows.get(source_index) if source_index is not None else None
        record: dict[str, Any] = {
            "length": len(vector),
            "min": min(vector) if vector else None,
            "max": max(vector) if vector else None,
            "nonzero": sum(value != 0 for value in vector),
            "unique_values": len(set(vector)),
            "source_index_from_filename": source_index,
        }
        if source:
            baseline = source["payload"]
            record.update(
                {
                    "source_label": source["label"],
                    "source_metadata": source["metadata"],
                    "exact_source_match": vector == baseline,
                    "source_multiset_match": Counter(vector) == Counter(baseline),
                    "positions_changed_from_source": sum(
                        left != right for left, right in zip(vector, baseline)
                    ),
                    "source_nonzero": sum(value != 0 for value in baseline),
                    "l1_change_from_source": sum(
                        abs(left - right) for left, right in zip(vector, baseline)
                    ),
                }
            )
        comparisons[name] = record
    return comparisons


def _source_text(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def _output_text(output: dict[str, Any]) -> str:
    pieces: list[str] = []
    text = output.get("text")
    if text:
        pieces.append("".join(text) if isinstance(text, list) else str(text))
    data = output.get("data") or {}
    for mime in ("text/plain", "text/markdown"):
        value = data.get(mime)
        if value:
            pieces.append("".join(value) if isinstance(value, list) else str(value))
    if output.get("output_type") == "error":
        pieces.append(f"{output.get('ename')}: {output.get('evalue')}")
    return "\n".join(pieces)


def _imports(source: str) -> Iterable[str]:
    cleaned = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("%", "!"))
    )
    try:
        tree = ast.parse(cleaned)
    except SyntaxError:
        return []
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def notebook_summary(path: Path) -> dict[str, Any]:
    notebook = json.loads(path.read_text(encoding="utf-8-sig"))
    cells = notebook.get("cells", [])
    execution_counts: list[int] = []
    output_counts: Counter[str] = Counter()
    imports: Counter[str] = Counter()
    file_literals: Counter[str] = Counter()
    landmarks: list[dict[str, Any]] = []
    output_previews: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for index, cell in enumerate(cells):
        source = _source_text(cell)
        if cell.get("cell_type") == "code":
            value = cell.get("execution_count")
            if isinstance(value, int):
                execution_counts.append(value)
            imports.update(_imports(source))
            file_literals.update(FILE_LITERAL.findall(source))
            matched_lines = [
                line.strip()
                for line in source.splitlines()
                if NOTEBOOK_LANDMARK.search(line)
            ]
            if matched_lines:
                landmarks.append({"cell": index, "lines": matched_lines[:30]})
            for output in cell.get("outputs", []):
                output_type = output.get("output_type", "unknown")
                output_counts[output_type] += 1
                text = _output_text(output).strip()
                if output_type == "error":
                    errors.append(
                        {
                            "cell": index,
                            "ename": output.get("ename"),
                            "evalue": output.get("evalue"),
                        }
                    )
                if text and len(output_previews) < 80:
                    output_previews.append({"cell": index, "text": text[:1200]})

    metadata = notebook.get("metadata", {})
    kernel = metadata.get("kernelspec") or {}
    languages = metadata.get("language_info") or {}
    return {
        "nbformat": f"{notebook.get('nbformat')}.{notebook.get('nbformat_minor')}",
        "kernel": kernel,
        "language": languages.get("name"),
        "cell_counts": dict(Counter(cell.get("cell_type", "unknown") for cell in cells)),
        "executed_code_cells": len(execution_counts),
        "execution_count_min": min(execution_counts) if execution_counts else None,
        "execution_count_max": max(execution_counts) if execution_counts else None,
        "output_counts": dict(output_counts),
        "errors": errors,
        "imports": sorted(imports),
        "file_literals": sorted(file_literals),
        "landmarks": landmarks,
        "output_previews": output_previews,
    }


def xlsx_summary(path: Path) -> dict[str, Any]:
    workbook_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    pkg_rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            element.attrib["Id"]: element.attrib["Target"]
            for element in relationships.findall(f"{{{pkg_rel_ns}}}Relationship")
        }
        sheets: list[dict[str, Any]] = []
        for sheet in workbook.findall(f".//{{{workbook_ns}}}sheet"):
            rel_id = sheet.attrib.get(f"{{{rel_ns}}}id")
            target = rel_targets.get(rel_id or "", "")
            if target.startswith("/"):
                member = target.lstrip("/")
            else:
                member = str(Path("xl") / target).replace("\\", "/")
            dimension = None
            if member in names:
                with archive.open(member) as stream:
                    for _event, element in ET.iterparse(stream, events=("start",)):
                        if element.tag == f"{{{workbook_ns}}}dimension":
                            dimension = element.attrib.get("ref")
                            break
                        if element.tag == f"{{{workbook_ns}}}sheetData":
                            break
            sheets.append(
                {
                    "name": sheet.attrib.get("name"),
                    "state": sheet.attrib.get("state", "visible"),
                    "member": member,
                    "dimension": dimension,
                    "compressed_bytes": archive.getinfo(member).compress_size if member in names else None,
                    "uncompressed_bytes": archive.getinfo(member).file_size if member in names else None,
                }
            )
        return {
            "sheets": sheets,
            "archive_members": len(names),
            "has_shared_strings": "xl/sharedStrings.xml" in names,
            "shared_strings_bytes": (
                archive.getinfo("xl/sharedStrings.xml").file_size
                if "xl/sharedStrings.xml" in names
                else 0
            ),
        }


def _xlsx_column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference.upper())
    if not letters:
        raise ValueError(f"Invalid cell reference: {reference}")
    result = 0
    for character in letters.group(0):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    member = "xl/sharedStrings.xml"
    if member not in archive.namelist():
        return []
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    root = ET.fromstring(archive.read(member))
    return [
        "".join(node.text or "" for node in item.findall(f".//{{{namespace}}}t"))
        for item in root.findall(f"{{{namespace}}}si")
    ]


def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str | None:
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    cell_type = cell.attrib.get("t")
    value = cell.find(f"{{{namespace}}}v")
    if value is None:
        inline = cell.find(f".//{{{namespace}}}t")
        return inline.text if inline is not None else None
    raw = value.text or ""
    if cell_type == "s":
        return shared_strings[int(raw)]
    if cell_type == "b":
        return "TRUE" if raw == "1" else "FALSE"
    return raw


def xlsx_numeric_profile(path: Path) -> dict[str, Any]:
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    structural = xlsx_summary(path)
    sheet = structural["sheets"][0]
    member = sheet["member"]
    digest_counts: Counter[str] = Counter()
    digest_labels: dict[str, Counter[str]] = {}
    label_counts: Counter[str] = Counter()
    header: list[str | None] = []
    first_data_rows: list[list[str | None]] = []
    feature_min: list[float] = []
    feature_max: list[float] = []
    nonfinite_cells = 0
    malformed_rows = 0
    data_rows = 0

    with zipfile.ZipFile(path) as archive:
        shared_strings = _xlsx_shared_strings(archive)
        with archive.open(member) as stream:
            for _event, row in ET.iterparse(stream, events=("end",)):
                if row.tag != f"{{{namespace}}}row":
                    continue
                cells: dict[int, str | None] = {}
                for cell in row.findall(f"{{{namespace}}}c"):
                    cells[_xlsx_column_index(cell.attrib["r"])] = _xlsx_cell_value(
                        cell, shared_strings
                    )
                width = max(cells, default=-1) + 1
                values = [cells.get(index) for index in range(width)]
                if not header:
                    header = values
                    feature_count = len(header) - (1 if header[-1] == "label" else 0)
                    feature_min = [float("inf")] * feature_count
                    feature_max = [float("-inf")] * feature_count
                    row.clear()
                    continue

                feature_count = len(feature_min)
                has_label = len(header) == feature_count + 1
                expected_width = feature_count + (1 if has_label else 0)
                if len(values) != expected_width or any(
                    value is None for value in values[:feature_count]
                ):
                    malformed_rows += 1
                    values.extend([None] * (expected_width - len(values)))
                if len(first_data_rows) < 3:
                    first_data_rows.append(values[:expected_width])

                digest = hashlib.sha256()
                for index, raw in enumerate(values[:feature_count]):
                    number = float(raw) if raw not in (None, "") else float("nan")
                    if number == 0:
                        number = 0.0
                    if number != number or number in (float("inf"), float("-inf")):
                        nonfinite_cells += 1
                    else:
                        if number < feature_min[index]:
                            feature_min[index] = number
                        if number > feature_max[index]:
                            feature_max[index] = number
                    digest.update(struct.pack("<d", number))
                digest_key = digest.hexdigest()
                digest_counts[digest_key] += 1
                if has_label:
                    label = str(values[feature_count])
                    label_counts[label] += 1
                    digest_labels.setdefault(digest_key, Counter())[label] += 1
                data_rows += 1
                row.clear()

    repeated = {key: count for key, count in digest_counts.items() if count > 1}
    conflicting = {
        key: labels
        for key, labels in digest_labels.items()
        if digest_counts[key] > 1 and len(labels) > 1
    }
    top_duplicates = [
        {
            "digest": key,
            "multiplicity": count,
            "labels": dict(digest_labels.get(key, {})),
        }
        for key, count in digest_counts.most_common(12)
        if count > 1
    ]
    constant_columns = [
        index for index, (minimum, maximum) in enumerate(zip(feature_min, feature_max))
        if minimum == maximum
    ]
    all_zero_columns = [
        index for index, (minimum, maximum) in enumerate(zip(feature_min, feature_max))
        if minimum == maximum == 0
    ]
    return {
        "sheet": sheet["name"],
        "header_first_12": header[:12],
        "header_last_12": header[-12:],
        "feature_count": len(feature_min),
        "data_rows": data_rows,
        "label_counts": dict(label_counts),
        "first_data_rows": first_data_rows,
        "malformed_rows": malformed_rows,
        "nonfinite_cells": nonfinite_cells,
        "constant_feature_columns": constant_columns,
        "all_zero_feature_columns": all_zero_columns,
        "unique_feature_rows": len(digest_counts),
        "duplicate_groups": len(repeated),
        "repeated_members": sum(repeated.values()),
        "redundant_rows": sum(count - 1 for count in repeated.values()),
        "max_multiplicity": max(digest_counts.values(), default=0),
        "conflicting_label_duplicate_groups": len(conflicting),
        "conflicting_label_repeated_members": sum(
            digest_counts[key] for key in conflicting
        ),
        "top_duplicate_groups": top_duplicates,
    }


def payload_row_digests(path: Path) -> tuple[list[str], list[str]]:
    digests: list[str] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        label_index = header.index("label")
        for row in reader:
            payload = bytes(int(float(value)) for value in row[:1500])
            digests.append(hashlib.sha256(payload).hexdigest())
            labels.append(row[label_index])
    return digests, labels


def xlsx_row_digests(path: Path) -> tuple[list[str], list[str]]:
    namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    structural = xlsx_summary(path)
    member = structural["sheets"][0]["member"]
    digests: list[str] = []
    labels: list[str] = []
    header: list[str | None] = []
    with zipfile.ZipFile(path) as archive:
        shared_strings = _xlsx_shared_strings(archive)
        with archive.open(member) as stream:
            for _event, row in ET.iterparse(stream, events=("end",)):
                if row.tag != f"{{{namespace}}}row":
                    continue
                cells = {
                    _xlsx_column_index(cell.attrib["r"]): _xlsx_cell_value(
                        cell, shared_strings
                    )
                    for cell in row.findall(f"{{{namespace}}}c")
                }
                width = max(cells, default=-1) + 1
                values = [cells.get(index) for index in range(width)]
                if not header:
                    header = values
                    row.clear()
                    continue
                has_label = header[-1] == "label"
                feature_count = len(header) - (1 if has_label else 0)
                digest = hashlib.sha256()
                for raw in values[:feature_count]:
                    number = float(raw) if raw not in (None, "") else float("nan")
                    if number == 0:
                        number = 0.0
                    digest.update(struct.pack("<d", number))
                digests.append(digest.hexdigest())
                if has_label:
                    labels.append(str(values[feature_count]))
                row.clear()
    return digests, labels


def raw_to_feature_collision_profile(
    payload_path: Path, feature_workbook_path: Path
) -> dict[str, Any]:
    raw_digests, raw_labels = payload_row_digests(payload_path)
    feature_digests, feature_labels = xlsx_row_digests(feature_workbook_path)
    if len(raw_digests) != len(feature_digests):
        raise ValueError("Payload and feature row counts differ")
    if feature_labels and raw_labels != feature_labels:
        first = next(
            index
            for index, (left, right) in enumerate(zip(raw_labels, feature_labels))
            if left != right
        )
        raise ValueError(f"Payload and feature label order differs at row {first}")

    raw_counts = Counter(raw_digests)
    feature_counts = Counter(feature_digests)
    feature_to_raw: dict[str, set[str]] = {}
    feature_to_labels: dict[str, Counter[str]] = {}
    raw_to_labels: dict[str, Counter[str]] = {}
    for raw, feature, label in zip(raw_digests, feature_digests, raw_labels):
        feature_to_raw.setdefault(feature, set()).add(raw)
        feature_to_labels.setdefault(feature, Counter())[label] += 1
        raw_to_labels.setdefault(raw, Counter())[label] += 1

    repeated_feature_members = sum(count for count in feature_counts.values() if count > 1)
    raw_first_members = sum(
        1
        for raw, feature in zip(raw_digests, feature_digests)
        if feature_counts[feature] > 1 and raw_counts[raw] > 1
    )
    post_raw_members = repeated_feature_members - raw_first_members
    raw_conflicting = {
        key: labels
        for key, labels in raw_to_labels.items()
        if raw_counts[key] > 1 and len(labels) > 1
    }
    feature_conflicting = {
        key: labels
        for key, labels in feature_to_labels.items()
        if feature_counts[key] > 1 and len(labels) > 1
    }
    multi_raw_feature_groups = {
        key: raw_set
        for key, raw_set in feature_to_raw.items()
        if feature_counts[key] > 1 and len(raw_set) > 1
    }
    return {
        "rows": len(raw_digests),
        "labels_aligned": (not feature_labels) or raw_labels == feature_labels,
        "raw_unique_rows": len(raw_counts),
        "raw_duplicate_groups": sum(count > 1 for count in raw_counts.values()),
        "raw_repeated_members": sum(count for count in raw_counts.values() if count > 1),
        "raw_redundant_rows": sum(count - 1 for count in raw_counts.values() if count > 1),
        "raw_conflicting_label_groups": len(raw_conflicting),
        "raw_conflicting_label_repeated_members": sum(
            raw_counts[key] for key in raw_conflicting
        ),
        "feature_unique_rows": len(feature_counts),
        "feature_duplicate_groups": sum(count > 1 for count in feature_counts.values()),
        "feature_repeated_members": repeated_feature_members,
        "feature_redundant_rows": sum(
            count - 1 for count in feature_counts.values() if count > 1
        ),
        "feature_conflicting_label_groups": len(feature_conflicting),
        "feature_conflicting_label_repeated_members": sum(
            feature_counts[key] for key in feature_conflicting
        ),
        "repeated_feature_members_already_raw_repeats": raw_first_members,
        "repeated_feature_members_first_repeated_post_raw": post_raw_members,
        "raw_share_of_repeated_feature_members": (
            raw_first_members / repeated_feature_members if repeated_feature_members else 0
        ),
        "post_raw_share_of_repeated_feature_members": (
            post_raw_members / repeated_feature_members if repeated_feature_members else 0
        ),
        "feature_duplicate_groups_combining_distinct_raw_payloads": len(
            multi_raw_feature_groups
        ),
        "max_distinct_raw_payloads_in_one_feature_group": max(
            (len(raw_set) for raw_set in multi_raw_feature_groups.values()), default=0
        ),
    }


def poisoned_vector_group_profile(vectors: dict[str, list[int]]) -> list[dict[str, Any]]:
    groups: dict[int, list[tuple[str, list[int]]]] = {}
    for name, vector in vectors.items():
        match = re.match(r"final_payload_(\d+)-", name)
        if match:
            groups.setdefault(int(match.group(1)), []).append((name, vector))
    output: list[dict[str, Any]] = []
    for source_index, members in sorted(groups.items()):
        pairwise: list[dict[str, Any]] = []
        for left_index in range(len(members)):
            for right_index in range(left_index + 1, len(members)):
                left_name, left = members[left_index]
                right_name, right = members[right_index]
                pairwise.append(
                    {
                        "left": left_name,
                        "right": right_name,
                        "positions_different": sum(
                            left_value != right_value
                            for left_value, right_value in zip(left, right)
                        ),
                        "multiset_match": Counter(left) == Counter(right),
                    }
                )
        output.append(
            {
                "source_index": source_index,
                "variants": [name for name, _vector in members],
                "pairwise": pairwise,
            }
        )
    return output


def zip_summary(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        return {
            "members": [
                {
                    "name": info.filename,
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                }
                for info in archive.infolist()
            ]
        }


def compare_zip_to_extracted(zip_path: Path, source_dir: Path) -> dict[str, Any]:
    """Check whether ZIP members add anything beyond extracted peer files."""
    members: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            member_bytes = archive.read(info)
            extracted_path = source_dir / Path(info.filename).name
            extracted_exists = extracted_path.is_file()
            extracted_hash = sha256(extracted_path) if extracted_exists else None
            member_hash = hashlib.sha256(member_bytes).hexdigest()
            members.append(
                {
                    "member": info.filename,
                    "member_sha256": member_hash,
                    "extracted_path": str(extracted_path),
                    "extracted_exists": extracted_exists,
                    "extracted_sha256": extracted_hash,
                    "exact_match": extracted_hash == member_hash,
                }
            )
    return {
        "zip_path": str(zip_path),
        "member_count": len(members),
        "all_members_have_extracted_peer": all(
            member["extracted_exists"] for member in members
        ),
        "all_extracted_peers_match_exactly": all(
            member["exact_match"] for member in members
        ),
        "members": members,
    }


def audit(source_dir: Path, repo_root: Path, deep: bool = False) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in source_dir.rglob("*") if item.is_file()):
        record: dict[str, Any] = {
            "path": str(path),
            "relative_path": str(path.relative_to(source_dir)),
            "extension": path.suffix.lower(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        if path.suffix.lower() == ".ipynb":
            record["notebook"] = notebook_summary(path)
        elif path.suffix.lower() == ".csv":
            record["csv"] = csv_summary(path)
        elif path.suffix.lower() == ".xlsx":
            record["xlsx"] = xlsx_summary(path)
        elif path.suffix.lower() == ".zip":
            record["zip"] = zip_summary(path)
        files.append(record)

    comparisons: dict[str, Any] = {}
    supplied_payload = source_dir / "Copy of Payload_data_UNSW.csv"
    repository_payload = repo_root / "data" / "Payload_data_UNSW.csv"
    if supplied_payload.exists() and repository_payload.exists():
        comparisons["payload_dataset"] = {
            "supplied_path": str(supplied_payload),
            "repository_path": str(repository_payload),
            "supplied_bytes": supplied_payload.stat().st_size,
            "repository_bytes": repository_payload.stat().st_size,
            "supplied_sha256": next(
                record["sha256"] for record in files if Path(record["path"]) == supplied_payload
            ),
            "repository_sha256": sha256(repository_payload),
        }
        comparisons["payload_dataset"]["exact_match"] = (
            comparisons["payload_dataset"]["supplied_sha256"]
            == comparisons["payload_dataset"]["repository_sha256"]
        )
        comparisons["payload_dataset"]["semantic_comparison"] = compare_payload_csvs(
            supplied_payload, repository_payload
        )

    if supplied_payload.exists():
        vectors = poisoned_payload_vectors(source_dir)
        profile = payload_dataset_profile(supplied_payload, vectors)
        comparisons["supplied_payload_profile"] = {
            key: value for key, value in profile.items() if key != "named_source_rows"
        }
        comparisons["poisoned_payload_vectors"] = compare_poisoned_vectors(vectors, profile)
        comparisons["poisoned_payload_variant_groups"] = poisoned_vector_group_profile(vectors)

    zip_paths = sorted(source_dir.glob("*.zip"))
    if zip_paths:
        comparisons["zip_to_extracted"] = [
            compare_zip_to_extracted(path, source_dir) for path in zip_paths
        ]

    for workbook_name in ("tda_280_x_y.xlsx", "tda_begnin_X_126.xlsx"):
        workbook_path = source_dir / workbook_name
        if workbook_path.exists():
            comparisons.setdefault("xlsx_numeric_profiles", {})[workbook_name] = (
                xlsx_numeric_profile(workbook_path)
            )

    if deep and supplied_payload.exists():
        full_feature_path = source_dir / "tda_280_x_y.xlsx"
        if full_feature_path.exists():
            comparisons["raw_to_280_feature_collisions"] = (
                raw_to_feature_collision_profile(supplied_payload, full_feature_path)
            )

    return {
        "source_dir": str(source_dir),
        "file_count": len(files),
        "extension_counts": dict(Counter(record["extension"] for record in files)),
        "comparisons": comparisons,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("monkam_files"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Also trace row-aligned raw-payload collisions into the 280-feature workbook.",
    )
    args = parser.parse_args()

    source_dir = args.source.resolve()
    repo_root = args.repo_root.resolve()
    report = audit(source_dir, repo_root, deep=args.deep)
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        sys.stdout.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
