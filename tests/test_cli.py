import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
from types import SimpleNamespace
from unittest.mock import patch

import httpx2
from openai import AuthenticationError, NotFoundError
from PIL import Image

from photo_tagger import (
    __version__,
    build_parser,
    main,
    normalize_location,
    run,
    validate_args,
)
from tagger.categories import CATEGORIES, CATEGORY_IDS, CATEGORY_RULES
from tagger.errors import FatalError
from tagger.image_loader import Photo, _format_exif_value, extract_exif, load_photo
from tagger.metadata import normalize_keywords, normalize_metadata, truncate_text
from tagger.openai_client import (
    DEFAULT_MODEL,
    MetadataGenerator,
    parse_response_content,
    resolve_api_key,
)
from tagger.output import build_document, write_document, write_text
from tagger.prompts import (
    build_response_schema,
    build_system_prompt,
    build_user_prompt,
    describe_orientation,
)
from tagger.tagging import collect_photo_paths, tag_photo, tag_photos
from tagger.targets import GALLERY, GALLERY_PROFILE, STOCK, STOCK_PROFILE, get_profile


def make_photo(path: str = "photo.jpg", **overrides) -> Photo:
    """Build a Photo that does not touch the file system."""
    values = {
        "path": path,
        "data_url": "data:image/jpeg;base64,AAAA",
        "width": 3000,
        "height": 2000,
        "exif": {},
    }
    values.update(overrides)
    return Photo(**values)


def write_test_image(path: str, size=(64, 48), color=(120, 90, 60)) -> str:
    """Write a small JPEG to disk and return its path."""
    Image.new("RGB", size, color).save(path, format="JPEG")
    return path


class FakeGenerator:
    """Stand-in for MetadataGenerator that records its calls."""

    def __init__(self, metadata=None, error=None):
        self.metadata = metadata or {
            "title": "Generated title",
            "description": "Generated description",
            "keywords": ["one", "two"],
        }
        self.error = error
        self.calls = []

    def generate(self, photo, profile):
        self.calls.append((photo.path, profile.name))
        if self.error is not None:
            raise self.error
        return dict(self.metadata)


def make_api_error(error_class, status: int, message: str):
    """Build an OpenAI status error without performing a request."""
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx2.Response(status, request=request)
    return error_class(message, response=response, body=None)


def make_completion(content=None, refusal=None, finish_reason="stop"):
    """Build an object shaped like a chat completion response."""
    message = SimpleNamespace(content=content, refusal=refusal)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


class FakeClient:
    """Stand-in for the OpenAI client that returns or raises a fixed value."""

    def __init__(self, response=None, error=None):
        self.requests = []

        def create(**kwargs):
            self.requests.append(kwargs)
            if error is not None:
                raise error
            return response

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


VALID_CONTENT = json.dumps(
    {
        "title": "Title",
        "description": "Description",
        "keywords": ["sky"],
        "primary_category": "landscape",
        "secondary_category": None,
    }
)

CATEGORY_CONTENT = {
    "title": "t",
    "description": "d",
    "keywords": ["k"],
    "primary_category": "seascape",
    "secondary_category": None,
}

MADEIRA = "Madeira, Portugal"


def parse_args(*argv, target=STOCK, output="out.json", location=None):
    """Parse tagging arguments with every required flag already filled in."""
    required = ["-t", target, "-o", output]
    required += ["-l", location] if location else ["--no-location"]
    return build_parser().parse_args([*required, *argv])


def assert_usage_error(test, argv) -> str:
    """Assert that argparse rejects the arguments and return its message."""
    stderr = StringIO()
    with redirect_stderr(stderr), test.assertRaises(SystemExit) as ctx:
        build_parser().parse_args(argv)
    test.assertEqual(ctx.exception.code, 2)
    return stderr.getvalue()


class ParserTests(unittest.TestCase):
    def test_parser_accepts_multiple_files(self):
        args = parse_args("a.jpg", "b.png", "c.tif")

        self.assertEqual(args.files, ["a.jpg", "b.png", "c.tif"])

    def test_required_flags_are_parsed(self):
        args = build_parser().parse_args(
            ["-t", "gallery", "-o", "labels.json", "-l", MADEIRA, "a.jpg"]
        )

        self.assertEqual(args.target, GALLERY)
        self.assertEqual(args.output, "labels.json")
        self.assertEqual(args.location, MADEIRA)
        self.assertFalse(args.no_location)

    def test_long_flags_are_accepted(self):
        args = build_parser().parse_args(
            ["--target", "stock", "--output", "-", "--no-location", "a.jpg"]
        )

        self.assertEqual(args.target, STOCK)
        self.assertEqual(args.output, "-")
        self.assertIsNone(args.location)
        self.assertTrue(args.no_location)

    def test_target_is_required(self):
        message = assert_usage_error(self, ["-o", "out.json", "--no-location", "a.jpg"])

        self.assertIn("--target", message)

    def test_output_is_required(self):
        message = assert_usage_error(self, ["-t", "stock", "--no-location", "a.jpg"])

        self.assertIn("--output", message)

    def test_location_or_no_location_is_required(self):
        message = assert_usage_error(self, ["-t", "stock", "-o", "out.json", "a.jpg"])

        self.assertIn("--location", message)
        self.assertIn("--no-location", message)

    def test_location_and_no_location_are_exclusive(self):
        message = assert_usage_error(
            self,
            ["-t", "stock", "-o", "o.json", "-l", MADEIRA, "--no-location", "a.jpg"],
        )

        self.assertIn("not allowed with", message)

    def test_files_are_required(self):
        assert_usage_error(self, ["-t", "stock", "-o", "out.json", "--no-location"])

    def test_unknown_target_is_rejected(self):
        assert_usage_error(
            self, ["-t", "postcard", "-o", "out.json", "--no-location", "a.jpg"]
        )

    def test_force_defaults_to_false(self):
        self.assertFalse(parse_args("a.jpg").force)
        self.assertTrue(parse_args("--force", "a.jpg").force)

    def test_help_lists_required_arguments_separately(self):
        stdout = StringIO()
        with redirect_stdout(stdout), self.assertRaises(SystemExit):
            build_parser().parse_args(["--help"])

        help_text = stdout.getvalue()
        required = help_text[help_text.index("required arguments:") :]
        for flag in ("--target", "--output", "--location", "--no-location"):
            self.assertIn(flag, required)
        self.assertNotIn("--model", required)

    def test_version_flag_is_available(self):
        stdout = StringIO()
        with redirect_stdout(stdout), self.assertRaises(SystemExit):
            build_parser().parse_args(["--version"])

        self.assertIn(__version__, stdout.getvalue())

    @patch.dict(os.environ, {"PHOTO_TAGGER_MODEL": ""}, clear=False)
    def test_model_defaults_to_builtin_model(self):
        self.assertEqual(parse_args("a.jpg").model, DEFAULT_MODEL)

    @patch.dict(os.environ, {"PHOTO_TAGGER_MODEL": "gpt-custom"}, clear=False)
    def test_model_default_is_read_from_environment(self):
        self.assertEqual(parse_args("a.jpg").model, "gpt-custom")

    @patch.dict(os.environ, {"PHOTO_TAGGER_MODEL": "gpt-custom"}, clear=False)
    def test_model_flag_overrides_environment(self):
        self.assertEqual(parse_args("--model", "gpt-flag", "a.jpg").model, "gpt-flag")

    def test_detail_defaults_to_high(self):
        self.assertEqual(parse_args("a.jpg").detail, "high")

    def test_unknown_detail_is_rejected(self):
        assert_usage_error(
            self,
            ["-t", "stock", "-o", "o.json", "--no-location", "--detail", "ultra", "a"],
        )


class ArgumentValidationTests(unittest.TestCase):
    def _args(self, **overrides):
        args = parse_args("a.jpg")
        for name, value in overrides.items():
            setattr(args, name, value)
        return args

    def test_max_dimension_below_minimum_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_args(self._args(max_dimension=32))
        self.assertIn("--max-dimension", str(ctx.exception))

    def test_non_positive_timeout_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_args(self._args(timeout=0))
        self.assertIn("--timeout", str(ctx.exception))

    def test_negative_retries_are_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_args(self._args(retries=-1))
        self.assertIn("--retries", str(ctx.exception))

    def test_zero_jobs_are_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_args(self._args(jobs=0))
        self.assertIn("--jobs", str(ctx.exception))

    def test_empty_model_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_args(self._args(model=" "))
        self.assertIn("--model", str(ctx.exception))

    def test_blank_location_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_args(self._args(location="   "))
        self.assertIn("--location", str(ctx.exception))

    def test_location_whitespace_is_collapsed(self):
        self.assertEqual(normalize_location("  Madeira,\n  Portugal "), MADEIRA)
        self.assertIsNone(normalize_location(None))

    def test_valid_arguments_pass(self):
        # Should not raise
        validate_args(self._args())


class TargetProfileTests(unittest.TestCase):
    def test_stock_profile_is_tuned_for_search(self):
        self.assertEqual(get_profile(STOCK), STOCK_PROFILE)
        self.assertGreaterEqual(STOCK_PROFILE.min_keywords, 25)
        self.assertLessEqual(STOCK_PROFILE.max_keywords, 50)

    def test_gallery_profile_keeps_descriptions_short(self):
        self.assertEqual(get_profile(GALLERY), GALLERY_PROFILE)
        self.assertLess(
            GALLERY_PROFILE.description_max_chars,
            STOCK_PROFILE.description_max_chars,
        )

    def test_unknown_profile_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get_profile("postcard")
        self.assertIn("Unknown target", str(ctx.exception))


class PromptTests(unittest.TestCase):
    def test_stock_prompt_asks_for_search_friendly_text(self):
        prompt = build_system_prompt(STOCK_PROFILE)

        self.assertIn("stock photography metadata specialist", prompt)
        self.assertIn(str(STOCK_PROFILE.title_max_chars), prompt)
        self.assertIn(str(STOCK_PROFILE.max_keywords), prompt)

    def test_gallery_prompt_asks_for_a_wall_label(self):
        prompt = build_system_prompt(GALLERY_PROFILE)

        self.assertIn("museum curator", prompt)
        self.assertIn("wall label", prompt)
        self.assertIn("single sentence of no more than twenty words", prompt)
        self.assertIn("single exhibition room", prompt)

    def test_profiles_produce_different_prompts(self):
        self.assertNotEqual(
            build_system_prompt(STOCK_PROFILE),
            build_system_prompt(GALLERY_PROFILE),
        )

    def test_system_prompt_lists_every_category_and_rule(self):
        prompt = build_system_prompt(STOCK_PROFILE)

        self.assertIn("primary_category", prompt)
        self.assertIn("secondary_category", prompt)
        for category in CATEGORIES:
            self.assertIn(f"- {category.id}: {category.description}", prompt)
        for rule in CATEGORY_RULES:
            self.assertIn(rule, prompt)

    def test_every_target_asks_for_categories(self):
        for profile in (STOCK_PROFILE, GALLERY_PROFILE):
            with self.subTest(target=profile.name):
                self.assertIn("Categories:", build_system_prompt(profile))
                self.assertIn(
                    "keywords and categories",
                    build_user_prompt(make_photo(), profile),
                )

    def test_schema_requires_a_primary_and_a_nullable_secondary_category(self):
        schema = build_response_schema(STOCK_PROFILE)["schema"]

        self.assertEqual(
            schema["required"],
            [
                "title",
                "description",
                "keywords",
                "primary_category",
                "secondary_category",
            ],
        )
        primary = schema["properties"]["primary_category"]
        secondary = schema["properties"]["secondary_category"]
        self.assertEqual(primary["type"], "string")
        self.assertEqual(primary["enum"], list(CATEGORY_IDS))
        self.assertEqual(secondary["type"], ["string", "null"])
        self.assertEqual(secondary["enum"], [*CATEGORY_IDS, None])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))

    def test_system_prompt_states_the_location_rules(self):
        prompt = build_system_prompt(STOCK_PROFILE, MADEIRA)

        self.assertIn("taken in Madeira, Portugal", prompt)
        self.assertIn("never contradict", prompt)
        self.assertIn("specific_place", prompt)
        self.assertIn("at most four location keywords", prompt)

    def test_system_prompt_without_location_has_no_location_rules(self):
        prompt = build_system_prompt(STOCK_PROFILE)

        self.assertNotIn("Location:", prompt)
        self.assertNotIn("specific_place", prompt)

    def test_user_prompt_asks_for_the_specific_place_with_a_location(self):
        photo = make_photo()

        self.assertIn(
            "keywords, categories and the specific place",
            build_user_prompt(photo, STOCK_PROFILE, MADEIRA),
        )
        self.assertIn(
            "a title, a description, keywords and categories for",
            build_user_prompt(photo, STOCK_PROFILE),
        )

    def test_schema_requires_a_nullable_specific_place_with_a_location(self):
        schema = build_response_schema(STOCK_PROFILE, MADEIRA)["schema"]

        self.assertIn("specific_place", schema["required"])
        self.assertEqual(
            schema["properties"]["specific_place"]["type"], ["string", "null"]
        )
        self.assertIn(MADEIRA, schema["properties"]["specific_place"]["description"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))

    def test_schema_without_location_has_no_specific_place(self):
        schema = build_response_schema(STOCK_PROFILE)["schema"]

        self.assertNotIn("specific_place", schema["properties"])

    def test_user_prompt_includes_exif_context(self):
        photo = make_photo(exif={"Model": "Leica M11", "FNumber": "f/2"})

        prompt = build_user_prompt(photo, STOCK_PROFILE)

        self.assertIn("Leica M11", prompt)
        self.assertIn("f/2", prompt)
        self.assertIn("photo.jpg", prompt)

    def test_orientation_is_derived_from_dimensions(self):
        self.assertEqual(describe_orientation(3000, 2000), "landscape")
        self.assertEqual(describe_orientation(2000, 3000), "portrait")
        self.assertEqual(describe_orientation(2000, 2000), "square")
        self.assertIsNone(describe_orientation(0, 0))


class MetadataNormalizationTests(unittest.TestCase):
    def test_truncate_text_keeps_short_text(self):
        self.assertEqual(truncate_text("A short title", 70), "A short title")

    def test_truncate_text_cuts_on_word_boundary(self):
        self.assertEqual(truncate_text("one two three four", 12), "one two...")

    def test_truncate_text_collapses_whitespace(self):
        self.assertEqual(truncate_text("  one\n two  ", 70), "one two")

    def test_keywords_are_lowercased_and_deduplicated(self):
        keywords = normalize_keywords(["Sky", "sky", " SKY ", "Cloud"], 10)

        self.assertEqual(keywords, ["sky", "cloud"])

    def test_keywords_are_capped_at_the_profile_limit(self):
        keywords = normalize_keywords([f"keyword{index}" for index in range(80)], 49)

        self.assertEqual(len(keywords), 49)

    def test_non_string_keywords_are_skipped(self):
        self.assertEqual(normalize_keywords(["sky", 42, None, ""], 10), ["sky"])

    def test_normalize_metadata_applies_profile_limits(self):
        raw = {
            **CATEGORY_CONTENT,
            "title": "word " * 40,
            "description": "sentence " * 100,
            "keywords": ["sky", "sky", "cloud"],
        }

        metadata = normalize_metadata(raw, STOCK_PROFILE)

        self.assertLessEqual(len(metadata["title"]), STOCK_PROFILE.title_max_chars)
        self.assertLessEqual(
            len(metadata["description"]), STOCK_PROFILE.description_max_chars
        )
        self.assertEqual(metadata["keywords"], ["sky", "cloud"])

    def test_normalize_metadata_requires_a_title(self):
        with self.assertRaises(ValueError) as ctx:
            normalize_metadata({"description": "d", "keywords": ["k"]}, STOCK_PROFILE)
        self.assertIn("title", str(ctx.exception))

    def test_normalize_metadata_keeps_the_categories(self):
        raw = {
            **CATEGORY_CONTENT,
            "primary_category": "landmark",
            "secondary_category": "travel",
        }

        metadata = normalize_metadata(raw, STOCK_PROFILE)

        self.assertEqual(
            metadata["category"], {"primary": "landmark", "secondary": "travel"}
        )

    def test_normalize_metadata_accepts_no_secondary_category(self):
        metadata = normalize_metadata(CATEGORY_CONTENT, STOCK_PROFILE)

        self.assertEqual(
            metadata["category"], {"primary": "seascape", "secondary": None}
        )

    def test_normalize_metadata_drops_a_secondary_that_repeats_the_primary(self):
        raw = {**CATEGORY_CONTENT, "secondary_category": "seascape"}

        metadata = normalize_metadata(raw, STOCK_PROFILE)

        self.assertIsNone(metadata["category"]["secondary"])

    def test_normalize_metadata_rejects_an_invalid_primary_category(self):
        for primary in (None, "", "Seascape", "beach", 11, ["seascape"]):
            raw = {**CATEGORY_CONTENT, "primary_category": primary}
            with self.subTest(primary=primary), self.assertRaises(ValueError) as ctx:
                normalize_metadata(raw, STOCK_PROFILE)
            self.assertIn("unknown category", str(ctx.exception))

    def test_normalize_metadata_requires_a_primary_category(self):
        raw = dict(CATEGORY_CONTENT)
        del raw["primary_category"]

        with self.assertRaises(ValueError):
            normalize_metadata(raw, STOCK_PROFILE)

    def test_normalize_metadata_rejects_an_invalid_secondary_category(self):
        for secondary in ("", "beach", 11, ["travel"]):
            raw = {**CATEGORY_CONTENT, "secondary_category": secondary}
            with (
                self.subTest(secondary=secondary),
                self.assertRaises(ValueError) as ctx,
            ):
                normalize_metadata(raw, STOCK_PROFILE)
            self.assertIn("secondary category", str(ctx.exception))

    def test_category_list_is_well_formed(self):
        self.assertEqual(len(CATEGORY_IDS), len(set(CATEGORY_IDS)))
        for category in CATEGORIES:
            with self.subTest(category=category.id):
                self.assertRegex(category.id, r"^[a-z]+(_[a-z]+)*$")
                self.assertTrue(category.description.strip())

    def test_normalize_metadata_keeps_the_given_and_detected_place(self):
        raw = {**CATEGORY_CONTENT, "specific_place": "  Porto   Moniz "}

        metadata = normalize_metadata(raw, STOCK_PROFILE, location=MADEIRA)

        self.assertEqual(
            metadata["location"], {"given": MADEIRA, "detected": "Porto Moniz"}
        )

    def test_normalize_metadata_accepts_no_specific_place(self):
        for place in (None, "", "   "):
            raw = {**CATEGORY_CONTENT, "specific_place": place}
            with self.subTest(place=place):
                metadata = normalize_metadata(raw, STOCK_PROFILE, location=MADEIRA)
                self.assertEqual(
                    metadata["location"], {"given": MADEIRA, "detected": None}
                )

    def test_normalize_metadata_drops_a_repeat_of_the_given_location(self):
        for place in ("Madeira", "portugal", "Madeira, Portugal", "MADEIRA,  Portugal"):
            raw = {**CATEGORY_CONTENT, "specific_place": place}
            with self.subTest(place=place):
                metadata = normalize_metadata(raw, STOCK_PROFILE, location=MADEIRA)
                self.assertIsNone(metadata["location"]["detected"])

    def test_normalize_metadata_limits_the_specific_place_length(self):
        raw = {**CATEGORY_CONTENT, "specific_place": "Very long place " * 20}

        metadata = normalize_metadata(raw, STOCK_PROFILE, location=MADEIRA)

        self.assertLessEqual(len(metadata["location"]["detected"]), 100)

    def test_normalize_metadata_rejects_an_invalid_specific_place(self):
        for raw in (
            CATEGORY_CONTENT,
            {**CATEGORY_CONTENT, "specific_place": 42},
            {**CATEGORY_CONTENT, "specific_place": ["Porto Moniz"]},
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError) as ctx:
                normalize_metadata(raw, STOCK_PROFILE, location=MADEIRA)
            self.assertIn("specific place", str(ctx.exception))

    def test_normalize_metadata_without_location_has_no_location(self):
        raw = {**CATEGORY_CONTENT, "specific_place": "Porto Moniz"}

        self.assertNotIn("location", normalize_metadata(raw, STOCK_PROFILE))

    def test_normalize_metadata_requires_keywords(self):
        with self.assertRaises(ValueError) as ctx:
            normalize_metadata(
                {"title": "t", "description": "d", "keywords": []}, STOCK_PROFILE
            )
        self.assertIn("keywords", str(ctx.exception))


class ResponseParsingTests(unittest.TestCase):
    def test_valid_response_is_parsed(self):
        content = VALID_CONTENT

        metadata = parse_response_content(content, GALLERY_PROFILE)

        self.assertEqual(metadata["title"], "Title")
        self.assertEqual(metadata["keywords"], ["sky"])

    def test_empty_response_raises(self):
        with self.assertRaises(ValueError) as ctx:
            parse_response_content("", GALLERY_PROFILE)
        self.assertIn("empty", str(ctx.exception))

    def test_malformed_json_raises(self):
        with self.assertRaises(ValueError) as ctx:
            parse_response_content("{not json", GALLERY_PROFILE)
        self.assertIn("malformed JSON", str(ctx.exception))


class MetadataGeneratorTests(unittest.TestCase):
    def test_generate_returns_normalized_metadata(self):
        client = FakeClient(response=make_completion(VALID_CONTENT))
        generator = MetadataGenerator(model="gpt-test", detail="low", client=client)

        metadata = generator.generate(make_photo(), STOCK_PROFILE)

        self.assertEqual(metadata["title"], "Title")
        request = client.requests[0]
        self.assertEqual(request["model"], "gpt-test")
        image_part = request["messages"][1]["content"][1]
        self.assertEqual(image_part["image_url"]["detail"], "low")

    def test_generate_always_requests_the_category(self):
        client = FakeClient(response=make_completion(VALID_CONTENT))
        generator = MetadataGenerator(client=client)

        metadata = generator.generate(make_photo(), GALLERY_PROFILE)

        self.assertEqual(
            metadata["category"], {"primary": "landscape", "secondary": None}
        )
        request = client.requests[0]
        schema = request["response_format"]["json_schema"]["schema"]
        self.assertIn("primary_category", schema["required"])
        self.assertIn("- landscape:", request["messages"][0]["content"])

    def test_generate_rejects_a_missing_category(self):
        content = json.dumps(
            {"title": "Title", "description": "Description", "keywords": ["sky"]}
        )
        client = FakeClient(response=make_completion(content))
        generator = MetadataGenerator(client=client)

        with self.assertRaises(ValueError) as ctx:
            generator.generate(make_photo(), STOCK_PROFILE)
        self.assertIn("category", str(ctx.exception))

    def test_generate_sends_and_returns_the_location(self):
        content = json.dumps(
            {**json.loads(VALID_CONTENT), "specific_place": "Pico do Arieiro"}
        )
        client = FakeClient(response=make_completion(content))
        generator = MetadataGenerator(location=MADEIRA, client=client)

        metadata = generator.generate(make_photo(), STOCK_PROFILE)

        self.assertEqual(
            metadata["location"], {"given": MADEIRA, "detected": "Pico do Arieiro"}
        )
        self.assertEqual(metadata["category"]["primary"], "landscape")
        request = client.requests[0]
        self.assertIn("taken in Madeira, Portugal", request["messages"][0]["content"])
        schema = request["response_format"]["json_schema"]["schema"]
        self.assertIn("specific_place", schema["required"])

    def test_generate_without_location_does_not_ask_for_a_place(self):
        client = FakeClient(response=make_completion(VALID_CONTENT))
        generator = MetadataGenerator(client=client)

        metadata = generator.generate(make_photo(), STOCK_PROFILE)

        self.assertNotIn("location", metadata)
        request = client.requests[0]
        self.assertNotIn("Location:", request["messages"][0]["content"])

    def test_refusal_is_reported(self):
        client = FakeClient(response=make_completion(refusal="cannot help"))
        generator = MetadataGenerator(client=client)

        with self.assertRaises(ValueError) as ctx:
            generator.generate(make_photo(), STOCK_PROFILE)
        self.assertIn("refused", str(ctx.exception))

    def test_truncated_response_is_reported(self):
        client = FakeClient(
            response=make_completion('{"title": "Ti', finish_reason="length")
        )
        generator = MetadataGenerator(client=client)

        with self.assertRaises(ValueError) as ctx:
            generator.generate(make_photo(), STOCK_PROFILE)
        self.assertIn("cut short", str(ctx.exception))

    def test_unknown_model_is_fatal(self):
        error = make_api_error(NotFoundError, 404, "model not found")
        generator = MetadataGenerator(model="gpt-nope", client=FakeClient(error=error))

        with self.assertRaises(FatalError) as ctx:
            generator.generate(make_photo(), STOCK_PROFILE)
        self.assertIn("gpt-nope", str(ctx.exception))

    def test_rejected_api_key_is_fatal(self):
        error = make_api_error(AuthenticationError, 401, "invalid key")
        generator = MetadataGenerator(client=FakeClient(error=error))

        with self.assertRaises(FatalError):
            generator.generate(make_photo(), STOCK_PROFILE)


class ApiKeyTests(unittest.TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False)
    def test_missing_api_key_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            resolve_api_key(None)
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-from-environment"}, clear=False)
    def test_api_key_is_read_from_environment(self):
        self.assertEqual(resolve_api_key(None), "sk-from-environment")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-from-environment"}, clear=False)
    def test_explicit_api_key_wins(self):
        self.assertEqual(resolve_api_key("sk-explicit"), "sk-explicit")


class ImageLoaderTests(unittest.TestCase):
    def test_load_photo_encodes_a_data_url(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_test_image(os.path.join(directory, "sample.jpg"))

            photo = load_photo(path, max_dimension=32)

            self.assertTrue(photo.data_url.startswith("data:image/jpeg;base64,"))
            self.assertEqual((photo.width, photo.height), (64, 48))
            self.assertEqual(photo.filename, "sample.jpg")

    def test_load_photo_downscales_the_encoded_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_test_image(
                os.path.join(directory, "sample.jpg"), size=(400, 300)
            )

            photo = load_photo(path, max_dimension=64)

            import base64

            payload = base64.b64decode(photo.data_url.split(",", 1)[1])
            with Image.open(BytesIO(payload)) as encoded:
                self.assertLessEqual(max(encoded.size), 64)

    def test_missing_photo_raises(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            load_photo("does-not-exist.jpg")
        self.assertIn("Photo not found", str(ctx.exception))

    def test_unsupported_extension_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "notes.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("not an image")

            with self.assertRaises(ValueError) as ctx:
                load_photo(path)
            self.assertIn("Unsupported image format", str(ctx.exception))

    def test_corrupt_image_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "broken.jpg")
            with open(path, "wb") as handle:
                handle.write(b"not really a jpeg")

            with self.assertRaises(ValueError) as ctx:
                load_photo(path)
            self.assertIn("Unable to decode image", str(ctx.exception))

    def test_exif_is_skipped_when_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_test_image(os.path.join(directory, "sample.jpg"))

            photo = load_photo(path, include_exif=False)

            self.assertEqual(photo.exif, {})

    def test_exif_orientation_is_applied(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "rotated.jpg")
            exif = Image.Exif()
            exif[0x0112] = 6  # Orientation: rotate 90 degrees clockwise
            Image.new("RGB", (64, 48)).save(path, format="JPEG", exif=exif)

            photo = load_photo(path)

            self.assertEqual((photo.width, photo.height), (48, 64))

    def test_iso_stored_as_a_tuple_is_formatted(self):
        self.assertEqual(_format_exif_value("ISOSpeedRatings", (400,)), "ISO 400")

    def test_extract_exif_returns_empty_without_metadata(self):
        self.assertEqual(extract_exif(Image.new("RGB", (8, 8))), {})


class PathCollectionTests(unittest.TestCase):
    def test_files_are_returned_in_order(self):
        self.assertEqual(
            collect_photo_paths(["b.jpg", "a.jpg"]),
            ["b.jpg", "a.jpg"],
        )

    def test_duplicates_are_removed(self):
        self.assertEqual(collect_photo_paths(["a.jpg", "a.jpg"]), ["a.jpg"])

    def test_directories_are_expanded(self):
        with tempfile.TemporaryDirectory() as directory:
            write_test_image(os.path.join(directory, "b.jpg"))
            write_test_image(os.path.join(directory, "a.jpg"))
            with open(os.path.join(directory, "notes.txt"), "w", encoding="utf-8") as h:
                h.write("ignored")

            paths = collect_photo_paths([directory])

            self.assertEqual(
                [os.path.basename(path) for path in paths], ["a.jpg", "b.jpg"]
            )

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError) as ctx:
            collect_photo_paths([])
        self.assertIn("No photos given", str(ctx.exception))

    def test_directory_without_images_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as ctx:
                collect_photo_paths([directory])
            self.assertIn("No supported image files", str(ctx.exception))


class TaggingTests(unittest.TestCase):
    def test_tag_photo_returns_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_test_image(os.path.join(directory, "sample.jpg"))
            generator = FakeGenerator()

            result = tag_photo(path, GALLERY_PROFILE, generator)

            self.assertTrue(result.succeeded)
            self.assertEqual(result.metadata["title"], "Generated title")
            self.assertEqual(generator.calls, [(path, GALLERY)])

    def test_tag_photo_captures_loading_errors(self):
        result = tag_photo("missing.jpg", STOCK_PROFILE, FakeGenerator())

        self.assertFalse(result.succeeded)
        self.assertIn("Photo not found", result.error)

    def test_tag_photo_captures_api_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_test_image(os.path.join(directory, "sample.jpg"))
            generator = FakeGenerator(error=RuntimeError("rate limit reached"))

            result = tag_photo(path, STOCK_PROFILE, generator)

            self.assertFalse(result.succeeded)
            self.assertIn("rate limit", result.error)

    def test_one_failure_does_not_stop_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            good = write_test_image(os.path.join(directory, "good.jpg"))
            progress = []

            results = tag_photos(
                ["missing.jpg", good],
                STOCK_PROFILE,
                FakeGenerator(),
                on_progress=lambda index, total, result: progress.append(index),
            )

            self.assertEqual([result.succeeded for result in results], [False, True])
            self.assertEqual(progress, [1, 2])

    def test_fatal_error_stops_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [
                write_test_image(os.path.join(directory, f"{index}.jpg"))
                for index in range(3)
            ]
            generator = FakeGenerator(error=FatalError("invalid key"))

            with self.assertRaises(FatalError):
                tag_photos(paths, STOCK_PROFILE, generator)
            self.assertEqual(len(generator.calls), 1)

    def test_concurrent_run_keeps_input_order(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [
                write_test_image(os.path.join(directory, f"{index}.jpg"))
                for index in range(6)
            ]
            paths.insert(2, "missing.jpg")
            progress = []

            results = tag_photos(
                paths,
                STOCK_PROFILE,
                FakeGenerator(),
                on_progress=lambda index, total, result: progress.append(index),
                jobs=3,
            )

            self.assertEqual([result.path for result in results], paths)
            self.assertFalse(results[2].succeeded)
            self.assertEqual(sorted(progress), list(range(1, len(paths) + 1)))

    def test_fatal_error_stops_a_concurrent_run(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [
                write_test_image(os.path.join(directory, f"{index}.jpg"))
                for index in range(4)
            ]

            with self.assertRaises(FatalError):
                tag_photos(
                    paths,
                    STOCK_PROFILE,
                    FakeGenerator(error=FatalError("invalid key")),
                    jobs=2,
                )


class DocumentTests(unittest.TestCase):
    def test_document_lists_successful_photos(self):
        results = tag_photos(["missing.jpg"], STOCK_PROFILE, FakeGenerator())
        document = build_document(results, STOCK_PROFILE, "test-model")

        self.assertEqual(document["target"], STOCK)
        self.assertEqual(document["model"], "test-model")
        self.assertEqual(document["photos"], [])
        self.assertEqual(len(document["errors"]), 1)

    def test_document_omits_errors_when_everything_succeeded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_test_image(os.path.join(directory, "sample.jpg"))
            results = tag_photos([path], GALLERY_PROFILE, FakeGenerator())

            document = build_document(results, GALLERY_PROFILE, "test-model")

            self.assertNotIn("errors", document)
            self.assertEqual(document["photos"][0]["filename"], "sample.jpg")
            self.assertEqual(document["photos"][0]["keywords"], ["one", "two"])

    def test_document_stores_the_category(self):
        generator = FakeGenerator(
            metadata={
                "title": "t",
                "description": "d",
                "keywords": ["k"],
                "category": {"primary": "food", "secondary": "drinks"},
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_test_image(os.path.join(directory, "sample.jpg"))
            results = tag_photos([path], STOCK_PROFILE, generator)

        document = build_document(results, STOCK_PROFILE, "test-model")

        self.assertEqual(
            document["photos"][0]["category"],
            {"primary": "food", "secondary": "drinks"},
        )

    def test_document_stores_the_location(self):
        location = {"given": MADEIRA, "detected": None}
        generator = FakeGenerator(
            metadata={
                "title": "t",
                "description": "d",
                "keywords": ["k"],
                "location": location,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = write_test_image(os.path.join(directory, "sample.jpg"))
            results = tag_photos([path], STOCK_PROFILE, generator)

        document = build_document(results, STOCK_PROFILE, "test-model")

        self.assertEqual(document["photos"][0]["location"], location)
        self.assertNotIn("category", document["photos"][0])

    def test_write_text_keeps_line_endings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "rows.csv")
            write_text("a,b\r\nc,d\r\n", path)

            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), b"a,b\r\nc,d\r\n")

    def test_write_text_reports_unwritable_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            blocker = os.path.join(directory, "file")
            write_text("x", blocker)

            with self.assertRaises(RuntimeError):
                write_text("x", os.path.join(blocker, "nested.csv"))

    def test_write_document_to_standard_output(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            write_document({"photos": []}, "-", indent=0)

        self.assertEqual(json.loads(stdout.getvalue()), {"photos": []})
        self.assertTrue(stdout.getvalue().endswith("\n"))

    def test_write_document_creates_the_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "nested", "tags.json")
            write_document({"target": STOCK, "photos": []}, path)

            with open(path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["target"], STOCK)


class RunTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.directory = self._directory.name
        self.photo = write_test_image(os.path.join(self.directory, "sample.jpg"))
        self.output = os.path.join(self.directory, "tags.json")

    def tearDown(self):
        self._directory.cleanup()

    def _run(self, args, generator=None):
        """Run a parsed tagging command against a fake generator."""
        generator = generator or FakeGenerator()
        with patch("photo_tagger.MetadataGenerator", return_value=generator) as factory:
            code = run(args)
        return code, factory, generator

    def _read_output(self) -> dict:
        with open(self.output, encoding="utf-8") as handle:
            return json.load(handle)

    def test_run_writes_the_requested_target(self):
        args = parse_args(self.photo, "-q", target=GALLERY, output=self.output)

        code, _, _ = self._run(args)

        self.assertEqual(code, 0)
        document = self._read_output()
        self.assertEqual(document["target"], GALLERY)
        self.assertEqual(len(document["photos"]), 1)

    def test_run_passes_the_location_to_the_generator(self):
        args = parse_args(
            self.photo, "-q", output=self.output, location=" Madeira,  Portugal "
        )

        code, factory, _ = self._run(args)

        self.assertEqual(code, 0)
        self.assertEqual(factory.call_args.kwargs["location"], MADEIRA)

    def test_run_with_no_location_passes_none(self):
        args = parse_args(self.photo, "-q", output=self.output)

        _, factory, _ = self._run(args)

        self.assertIsNone(factory.call_args.kwargs["location"])

    def test_run_refuses_to_replace_an_existing_output(self):
        with open(self.output, "w", encoding="utf-8") as handle:
            handle.write("previous album")
        args = parse_args(self.photo, "-q", output=self.output)

        with self.assertRaises(ValueError) as ctx:
            self._run(args)

        self.assertIn("--force", str(ctx.exception))
        with open(self.output, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), "previous album")

    def test_existing_output_is_checked_before_any_api_call(self):
        with open(self.output, "w", encoding="utf-8") as handle:
            handle.write("previous album")
        args = parse_args(self.photo, "-q", output=self.output)
        generator = FakeGenerator()

        with self.assertRaises(ValueError):
            self._run(args, generator)

        self.assertEqual(generator.calls, [])

    def test_force_replaces_an_existing_output(self):
        with open(self.output, "w", encoding="utf-8") as handle:
            handle.write("previous album")
        args = parse_args(self.photo, "--force", "-q", output=self.output)

        code, _, _ = self._run(args)

        self.assertEqual(code, 0)
        self.assertEqual(len(self._read_output()["photos"]), 1)

    def test_directory_output_is_rejected_even_with_force(self):
        args = parse_args(self.photo, "--force", "-q", output=self.directory)

        with self.assertRaises(ValueError) as ctx:
            self._run(args)
        self.assertIn("directory", str(ctx.exception))

    def test_standard_output_needs_no_force(self):
        args = parse_args(self.photo, "-q", output="-")
        stdout = StringIO()

        with redirect_stdout(stdout):
            code, _, _ = self._run(args)

        self.assertEqual(code, 0)
        self.assertEqual(len(json.loads(stdout.getvalue())["photos"]), 1)

    def test_main_reports_an_existing_output_with_exit_code_2(self):
        with open(self.output, "w", encoding="utf-8") as handle:
            handle.write("previous album")
        stderr = StringIO()

        with redirect_stderr(stderr):
            code = main(["-t", "stock", "-o", self.output, "--no-location", self.photo])

        self.assertEqual(code, 2)
        self.assertIn("already exists", stderr.getvalue())

    def test_run_reports_failure_exit_code(self):
        args = parse_args("missing.jpg", "-q", output=self.output)

        code, _, _ = self._run(args)

        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
