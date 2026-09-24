import json
import os

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
)

from .categories import CATEGORY_SETS, CategorySet
from .errors import FatalError
from .image_loader import Photo
from .metadata import normalize_metadata
from .prompts import build_response_schema, build_system_prompt, build_user_prompt
from .targets import TargetProfile

DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_DETAIL = "high"
DETAIL_LEVELS = ("low", "high", "auto")
API_KEY_VARIABLE = "OPENAI_API_KEY"
MODEL_VARIABLE = "PHOTO_TAGGER_MODEL"


def resolve_default_model() -> str:
    """Return the model from the environment, or the built-in default."""
    return os.environ.get(MODEL_VARIABLE, "").strip() or DEFAULT_MODEL


def resolve_api_key(api_key: str | None = None) -> str:
    """Return the API key from the argument or the environment.

    Args:
        api_key: Key passed on the command line, may be None

    Returns:
        The resolved API key

    Raises:
        RuntimeError: If no key is available
    """
    resolved = api_key or os.environ.get(API_KEY_VARIABLE, "")
    resolved = resolved.strip()

    if not resolved:
        raise RuntimeError(
            "OpenAI API key is not configured.\n"
            f"Set the {API_KEY_VARIABLE} environment variable:\n"
            f"  export {API_KEY_VARIABLE}=sk-...\n"
            "or pass the key with --api-key."
        )

    return resolved


def build_messages(
    photo: Photo,
    profile: TargetProfile,
    detail: str = DEFAULT_DETAIL,
    category_sets: tuple[CategorySet, ...] = (),
) -> list[dict]:
    """Build the chat messages describing a single photo.

    Args:
        photo: The prepared photo
        profile: Target profile describing the destination
        detail: Image detail level requested from the model
        category_sets: Category lists the model has to choose from

    Returns:
        A list of chat messages ready to be sent to the model
    """
    return [
        {"role": "system", "content": build_system_prompt(profile, category_sets)},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": build_user_prompt(photo, profile, category_sets),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": photo.data_url, "detail": detail},
                },
            ],
        },
    ]


def parse_response_content(
    content: str | None,
    profile: TargetProfile,
    category_sets: tuple[CategorySet, ...] = (),
) -> dict:
    """Parse and normalize the JSON payload returned by the model.

    Args:
        content: Raw message content returned by the model
        profile: Target profile describing the destination
        category_sets: Category lists the model had to choose from

    Returns:
        Normalized metadata with title, description and keywords

    Raises:
        ValueError: If the payload is empty, not valid JSON or incomplete
    """
    if not content or not content.strip():
        raise ValueError("Model returned an empty response")

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned malformed JSON: {e}") from e

    return normalize_metadata(payload, profile, category_sets)


class MetadataGenerator:
    """Generates photo metadata through the OpenAI vision models."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        detail: str = DEFAULT_DETAIL,
        category_sets: tuple[CategorySet, ...] = CATEGORY_SETS,
        client=None,
    ):
        """Create a generator.

        Args:
            model: Vision capable model identifier
            api_key: API key, falls back to the environment when None
            timeout: Per request timeout in seconds
            max_retries: Number of automatic retries on transient failures
            detail: Image detail level, one of "low", "high" or "auto"
            category_sets: Category lists the model picks from for every
                photo, by default every list an export format needs
            client: Pre-built OpenAI client, mainly used by the tests

        Raises:
            RuntimeError: If no API key is available and no client was given
        """
        self.model = model
        self.detail = detail
        self.category_sets = category_sets
        self._client = client or OpenAI(
            api_key=resolve_api_key(api_key),
            timeout=timeout,
            max_retries=max_retries,
        )

    def generate(self, photo: Photo, profile: TargetProfile) -> dict:
        """Generate metadata for a single photo.

        Args:
            photo: The prepared photo
            profile: Target profile describing the destination

        Returns:
            Normalized metadata with title, description and keywords

        Raises:
            FatalError: If the failure would repeat for every other photo
            RuntimeError: If the API call fails
            ValueError: If the model response cannot be used
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=build_messages(
                    photo, profile, self.detail, self.category_sets
                ),
                response_format={
                    "type": "json_schema",
                    "json_schema": build_response_schema(profile, self.category_sets),
                },
            )
        except AuthenticationError as e:
            raise FatalError(
                f"OpenAI rejected the API key: {e}. "
                f"Check the {API_KEY_VARIABLE} environment variable."
            ) from e
        except PermissionDeniedError as e:
            raise FatalError(
                f"The API key has no access to model {self.model}: {e}"
            ) from e
        except NotFoundError as e:
            raise FatalError(
                f"Model {self.model} is not available: {e}. Check the --model option."
            ) from e
        except RateLimitError as e:
            raise RuntimeError(f"OpenAI rate limit reached: {e}") from e
        except APITimeoutError as e:
            raise RuntimeError(f"OpenAI request timed out for {photo.filename}") from e
        except APIConnectionError as e:
            raise RuntimeError(f"Unable to reach the OpenAI API: {e}") from e
        except APIStatusError as e:
            raise RuntimeError(
                f"OpenAI returned an error for {photo.filename}: {e}"
            ) from e
        except OpenAIError as e:
            raise RuntimeError(
                f"OpenAI request failed for {photo.filename}: {e}"
            ) from e

        if not response.choices:
            raise ValueError(f"Model returned no choices for {photo.filename}")

        choice = response.choices[0]
        refusal = getattr(choice.message, "refusal", None)
        if refusal:
            raise ValueError(f"Model refused to describe {photo.filename}: {refusal}")

        if choice.finish_reason == "length":
            raise ValueError(f"Model response for {photo.filename} was cut short")

        return parse_response_content(
            choice.message.content, profile, self.category_sets
        )
