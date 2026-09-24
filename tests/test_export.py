import csv
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

import photo_export
import photo_tagger
from tagger.categories import ADOBE_STOCK_CATEGORIES
from tagger.export import (
    ADOBE_STOCK,
    ADOBE_STOCK_COLUMNS,
    build_adobe_stock_row,
    find_category,
    find_duplicate_filenames,
    get_format,
    load_document,
    render_adobe_stock_csv,
)
from tagger.openai_client import MetadataGenerator

HEADER = list(ADOBE_STOCK_COLUMNS)


def make_entry(filename="lighthouse.jpg", **overrides) -> dict:
    """Build a photo entry shaped like the ones photo_tagger.py writes."""
    entry = {
        "file": f"shoot/{filename}",
        "filename": filename,
        "title": "Lighthouse on rocky coastline at sunrise",
        "description": "A white lighthouse on a headland.",
        "keywords": ["lighthouse", "coastline", "sunrise"],
        "categories": {"adobe_stock": 11},
    }
    entry.update(overrides)
    return entry


def make_document(*photos, **extra) -> dict:
    document = {
        "target": "stock",
        "model": "test-model",
        "generated_at": "2026-09-24T10:00:00+00:00",
        "photos": list(photos),
    }
    document.update(extra)
    return document


def write_json(directory: str, document, name: str = "tags.json") -> str:
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, ensure_ascii=False)
    return path


def read_rows(path: str) -> list[list[str]]:
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def export(argv) -> tuple[int, str, str]:
    """Run photo_export.main and capture its exit code, stdout and stderr."""
    stdout, stderr = StringIO(), StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = photo_export.main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


class LoadDocumentTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.directory = self._directory.name

    def tearDown(self):
        self._directory.cleanup()

    def test_valid_document_is_returned(self):
        path = write_json(self.directory, make_document(make_entry()))

        document = load_document(path)

        self.assertEqual(document["photos"][0]["filename"], "lighthouse.jpg")

    def test_the_bundled_samples_are_valid(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in ("tags.stock.json", "tags.gallery.json"):
            with self.subTest(sample=name):
                document = load_document(os.path.join(root, "samples", name))
                photo = document["photos"][0]
                self.assertEqual(find_category(photo, ADOBE_STOCK_CATEGORIES), 11)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_document(os.path.join(self.directory, "missing.json"))

    def test_malformed_json_raises(self):
        path = os.path.join(self.directory, "broken.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{not json")

        with self.assertRaises(ValueError) as ctx:
            load_document(path)
        self.assertIn("not valid JSON", str(ctx.exception))

    def test_document_without_photos_raises(self):
        for document in ({"target": "stock"}, [], {"photos": {}}):
            with self.subTest(document=document), self.assertRaises(ValueError):
                load_document(write_json(self.directory, document))

    def test_incomplete_photo_raises(self):
        cases = {
            "filename": make_entry(filename=""),
            "title": make_entry(title=None),
            "keywords": make_entry(keywords="sea, sky"),
        }
        for field, entry in cases.items():
            path = write_json(self.directory, make_document(entry))
            with self.subTest(field=field), self.assertRaises(ValueError) as ctx:
                load_document(path)
            self.assertIn(field, str(ctx.exception))

    def test_photo_that_is_not_an_object_raises(self):
        path = write_json(self.directory, make_document("lighthouse.jpg"))

        with self.assertRaises(ValueError):
            load_document(path)


class CategoryLookupTests(unittest.TestCase):
    def test_known_category_is_returned(self):
        self.assertEqual(find_category(make_entry(), ADOBE_STOCK_CATEGORIES), 11)

    def test_missing_or_invalid_category_is_none(self):
        for categories in (
            None,
            "11",
            {},
            {"adobe_stock": 0},
            {"adobe_stock": 22},
            {"adobe_stock": 11.0},
            {"adobe_stock": True},
            {"adobe_stock": [11]},
        ):
            entry = make_entry(categories=categories)
            with self.subTest(categories=categories):
                self.assertIsNone(find_category(entry, ADOBE_STOCK_CATEGORIES))


class AdobeStockRowTests(unittest.TestCase):
    def test_row_follows_the_adobe_stock_columns(self):
        self.assertEqual(
            build_adobe_stock_row(make_entry()),
            {
                "Filename": "lighthouse.jpg",
                "Title": "Lighthouse on rocky coastline at sunrise",
                "Keywords": "lighthouse, coastline, sunrise",
                "Category": 11,
                "Releases": "",
            },
        )

    def test_description_and_path_are_not_exported(self):
        row = build_adobe_stock_row(make_entry())

        self.assertNotIn("shoot/", " ".join(str(value) for value in row.values()))
        self.assertNotIn("headland", " ".join(str(value) for value in row.values()))

    def test_commas_inside_keywords_are_removed_before_deduplication(self):
        entry = make_entry(keywords=["sea coast", "sea, coast", "Harbour", 7])

        self.assertEqual(build_adobe_stock_row(entry)["Keywords"], "sea coast, harbour")

    def test_keywords_are_capped_at_49_after_deduplication(self):
        keywords = ["duplicate"] * 5 + [f"keyword{index}" for index in range(60)]

        row = build_adobe_stock_row(make_entry(keywords=keywords))

        exported = row["Keywords"].split(", ")
        self.assertEqual(len(exported), 49)
        self.assertEqual(exported[:2], ["duplicate", "keyword0"])

    def test_title_is_limited_to_200_characters(self):
        row = build_adobe_stock_row(make_entry(title="word " * 60))

        self.assertLessEqual(len(row["Title"]), 200)

    def test_missing_category_leaves_the_cell_empty(self):
        entry = make_entry()
        del entry["categories"]

        self.assertEqual(build_adobe_stock_row(entry)["Category"], "")


class AdobeStockCsvTests(unittest.TestCase):
    def test_csv_quotes_commas_and_keeps_unicode(self):
        entry = make_entry(filename="café.jpg", title="Café terrace, Paris, at dusk")

        text = render_adobe_stock_csv([entry])

        self.assertEqual(
            text,
            "Filename,Title,Keywords,Category,Releases\r\n"
            'café.jpg,"Café terrace, Paris, at dusk",'
            '"lighthouse, coastline, sunrise",11,\r\n',
        )

    def test_csv_round_trips_through_a_reader(self):
        entries = [make_entry("a.jpg"), make_entry("b.jpg", categories={})]

        rows = list(csv.reader(StringIO(render_adobe_stock_csv(entries))))

        self.assertEqual(rows[0], HEADER)
        self.assertEqual([row[0] for row in rows[1:]], ["a.jpg", "b.jpg"])
        self.assertEqual([row[3] for row in rows[1:]], ["11", ""])

    def test_empty_document_renders_only_the_header(self):
        self.assertEqual(
            render_adobe_stock_csv([]), "Filename,Title,Keywords,Category,Releases\r\n"
        )


class FormatRegistryTests(unittest.TestCase):
    def test_adobe_stock_is_registered(self):
        export_format = get_format(ADOBE_STOCK)

        self.assertEqual(export_format.extension, ".csv")
        self.assertIs(export_format.category_set, ADOBE_STOCK_CATEGORIES)
        self.assertTrue(export_format.unique_filenames)

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            get_format("xml")

    def test_duplicate_filenames_are_reported_once_in_order(self):
        photos = [make_entry(name) for name in ("b.jpg", "a.jpg", "b.jpg", "b.jpg")]
        photos.append(make_entry("a.jpg"))

        self.assertEqual(find_duplicate_filenames(photos), ["b.jpg", "a.jpg"])


class ExportCommandTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.directory = self._directory.name

    def tearDown(self):
        self._directory.cleanup()

    def test_format_defaults_to_adobe_stock(self):
        args = photo_export.build_parser().parse_args(["tags.json"])

        self.assertEqual(args.format, ADOBE_STOCK)
        self.assertIsNone(args.output)

    def test_unknown_format_is_rejected(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            photo_export.build_parser().parse_args(["-f", "xml", "tags.json"])

    def test_default_output_is_written_next_to_the_input(self):
        path = write_json(self.directory, make_document(make_entry()))

        code, _, stderr = export([path])

        output = os.path.join(self.directory, "tags.adobe-stock.csv")
        self.assertEqual(code, 0)
        self.assertEqual(read_rows(output)[1][0], "lighthouse.jpg")
        self.assertIn("Exported 1 photo(s) as adobe-stock", stderr)
        self.assertNotIn("Warning", stderr)

    def test_explicit_output_is_used(self):
        path = write_json(self.directory, make_document(make_entry()))
        output = os.path.join(self.directory, "nested", "adobe.csv")

        code, _, _ = export([path, "-o", output])

        self.assertEqual(code, 0)
        self.assertEqual(read_rows(output)[0], HEADER)

    def test_output_can_go_to_standard_output(self):
        path = write_json(self.directory, make_document(make_entry()))

        code, stdout, stderr = export([path, "-o", "-", "--quiet"])

        self.assertEqual(code, 0)
        self.assertEqual(list(csv.reader(StringIO(stdout)))[0], HEADER)
        self.assertEqual(stderr, "")

    def test_missing_categories_and_failed_photos_are_warned_about(self):
        uncategorized = make_entry("old.jpg")
        del uncategorized["categories"]
        document = make_document(
            make_entry(),
            uncategorized,
            errors=[{"file": "broken.jpg", "error": "unreadable"}],
        )
        path = write_json(self.directory, document)

        code, stdout, stderr = export([path, "-o", "-"])

        self.assertEqual(code, 0)
        self.assertEqual(len(list(csv.reader(StringIO(stdout)))), 3)
        self.assertIn("1 photo(s) failed tagging", stderr)
        self.assertIn("no Adobe Stock category", stderr)
        self.assertIn("old.jpg", stderr)

    def test_quiet_suppresses_warnings(self):
        entry = make_entry()
        del entry["categories"]
        path = write_json(self.directory, make_document(entry))

        _, _, stderr = export([path, "-o", "-", "-q"])

        self.assertEqual(stderr, "")

    def test_duplicate_filenames_fail_without_writing(self):
        document = make_document(make_entry("IMG_1.jpg"), make_entry("IMG_1.jpg"))
        path = write_json(self.directory, document)
        output = os.path.join(self.directory, "adobe.csv")

        code, _, stderr = export([path, "-o", output])

        self.assertEqual(code, 2)
        self.assertIn("IMG_1.jpg", stderr)
        self.assertFalse(os.path.exists(output))

    def test_missing_input_exits_with_2(self):
        code, _, stderr = export([os.path.join(self.directory, "missing.json")])

        self.assertEqual(code, 2)
        self.assertIn("not found", stderr)

    def test_invalid_input_exits_with_2(self):
        path = write_json(self.directory, {"photos": [{"filename": "a.jpg"}]})

        code, _, stderr = export([path])

        self.assertEqual(code, 2)
        self.assertIn("Error:", stderr)

    def test_unwritable_output_exits_with_1(self):
        path = write_json(self.directory, make_document(make_entry()))

        code, _, stderr = export([path, "-o", os.path.join(path, "adobe.csv")])

        self.assertEqual(code, 1)
        self.assertIn("Unable to write", stderr)


class TagThenExportTests(unittest.TestCase):
    """Tags photos once, then exports the JSON without another API call."""

    def test_tagged_json_converts_to_adobe_stock_csv(self):
        response = json.dumps(
            {
                "title": "Mountain lake under a clear sky",
                "description": "Calm alpine lake below mountain peaks.",
                "keywords": ["mountain", "lake", "alpine, lake", "sky"],
                "adobe_stock_category": 11,
            }
        )
        requests = []

        def create(**kwargs):
            requests.append(kwargs)
            message = SimpleNamespace(content=response, refusal=None)
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice])

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with tempfile.TemporaryDirectory() as directory:
            photo = os.path.join(directory, "lake.jpg")
            Image.new("RGB", (64, 48), (40, 90, 160)).save(photo, format="JPEG")
            tags = os.path.join(directory, "tags.json")

            generator = MetadataGenerator(client=client)
            with (
                patch("photo_tagger.MetadataGenerator", return_value=generator),
                redirect_stderr(StringIO()),
            ):
                self.assertEqual(photo_tagger.main([photo, "-o", tags, "-q"]), 0)

            with open(tags, encoding="utf-8") as handle:
                document = json.load(handle)
            self.assertEqual(document["photos"][0]["categories"], {"adobe_stock": 11})

            with patch.object(
                MetadataGenerator, "generate", side_effect=AssertionError
            ):
                code, _, _ = export([tags, "-q"])

            self.assertEqual(code, 0)
            rows = read_rows(os.path.join(directory, "tags.adobe-stock.csv"))

        self.assertEqual(len(requests), 1)
        self.assertEqual(
            rows,
            [
                HEADER,
                [
                    "lake.jpg",
                    "Mountain lake under a clear sky",
                    "mountain, lake, alpine lake, sky",
                    "11",
                    "",
                ],
            ],
        )


if __name__ == "__main__":
    unittest.main()
