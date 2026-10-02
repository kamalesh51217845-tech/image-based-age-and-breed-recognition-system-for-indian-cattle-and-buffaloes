import json
import logging
from io import BytesIO
from typing import Dict, Optional, Tuple

from .classifier import ALL_BREEDS, AGE_GROUPS

logger = logging.getLogger(__name__)


def _extract_json_object(text: str) -> str:
    text = (text or "").strip()

    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    return text[start : end + 1]


def _bytes_to_pil_image(image_content: bytes):
    from PIL import Image

    return Image.open(BytesIO(image_content)).convert("RGB")


def _normalize_animal_type(value: str) -> str:
    s = (value or "").strip().lower()
    if s in {"cattle", "cow", "bull", "ox"}:
        return "Cattle"
    if s in {"buffalo", "water buffalo", "waterbuffalo"}:
        return "Buffalo"
    return value


def _normalize_age_group(value: str) -> str:
    s = (value or "").strip().lower()
    mapping = {
        "calf": "Calf (0-6 months)",
        "young": "Young (6-18 months)",
        "adult": "Adult (18 months - 6 years)",
        "old": "Old (6+ years)",
    }
    for key, canonical in mapping.items():
        if s == key or s.startswith(f"{key} "):
            return canonical
    for canonical in AGE_GROUPS:
        if s and s in canonical.lower():
            return canonical
    return value


def _normalize_breed(value: str) -> str:
    s = (value or "").strip().lower()
    for b in ALL_BREEDS:
        if b.lower() == s:
            return b
    return value


class ExternalAIClassifier:
    """
    Classifier that uses Google Gemini for breed and age recognition.

    Supports either:
    - `google-genai` (new SDK, import path: `from google import genai`)
    - `google-generativeai` (older SDK, import path: `import google.generativeai as genai`)
    """

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key
        self.model_name = model_name

        self._backend: Optional[str] = None  # "google_genai" | "google_generativeai"
        self._client = None
        self._model = None

        if not api_key:
            logger.warning("No AI_API_KEY provided. ExternalAIClassifier will be disabled.")
            return

        # Prefer the new SDK if available.
        try:
            from google import genai as google_genai  # type: ignore
            from google.genai import types as google_genai_types  # noqa: F401

            self._backend = "google_genai"
            self._client = google_genai.Client(api_key=api_key)
            return
        except Exception:
            pass

        # Fallback to the older `google-generativeai` package (present in requirements.txt).
        try:
            import google.generativeai as google_generativeai  # type: ignore

            self._backend = "google_generativeai"
            google_generativeai.configure(api_key=api_key)
            self._model = google_generativeai.GenerativeModel(model_name)
            return
        except Exception as e:
            logger.warning(f"External AI SDK unavailable. ExternalAIClassifier disabled. Error: {e}")
            self._backend = None
            self._client = None
            self._model = None

    @property
    def enabled(self) -> bool:
        return self._backend is not None

    def _call_gemini(self, prompt: str, image_content: bytes) -> str:
        if not self.enabled:
            raise ValueError("AI client not configured.")

        if self._backend == "google_genai":
            from google.genai import types as google_genai_types  # type: ignore

            image_part = google_genai_types.Part.from_bytes(data=image_content, mime_type="image/jpeg")
            response = self._client.models.generate_content(  # type: ignore[union-attr]
                model=self.model_name,
                contents=[prompt, image_part],
            )
            return (response.text or "").strip()

        if self._backend == "google_generativeai":
            image = _bytes_to_pil_image(image_content)
            response = self._model.generate_content([prompt, image])  # type: ignore[union-attr]
            return (getattr(response, "text", "") or "").strip()

        raise ValueError("Unsupported AI backend.")

    def validate_animal(self, image_content: bytes) -> Tuple[bool, str]:
        """
        Dedicated animal validation — checks if image is a cow or buffalo.
        Returns (is_valid, detected_animal_name).
        """
        if not self.enabled:
            raise ValueError("AI API Key not configured.")

        prompt = """Look very carefully at this image.

Your ONLY job: decide if the animal in the image is a COW (cattle) or BUFFALO.

Rules:
- Cow and cattle are the same — domestic bovine animals (zebu, taurine, any breed).
- Buffalo means domestic water buffalo (Indian buffalo like Murrah, Surti, etc.).
- Any other animal (tiger, dog, horse, goat, sheep, cat, elephant, deer, lion, etc.) is NOT valid.
- If the image has no animal at all, it is NOT valid.

Respond ONLY with this exact JSON and nothing else:
{"is_cow_or_buffalo": true_or_false, "detected": "name of what you see"}

Examples:
{"is_cow_or_buffalo": true, "detected": "cow"}
{"is_cow_or_buffalo": false, "detected": "tiger"}
{"is_cow_or_buffalo": false, "detected": "dog"}
"""
        try:
            text = self._call_gemini(prompt, image_content)
            print(f"[VALIDATION RAW RESPONSE]: {text}")
            result = json.loads(_extract_json_object(text))
            is_valid = bool(result.get("is_cow_or_buffalo", False))
            detected = str(result.get("detected", "unknown animal"))
            print(f"[VALIDATION RESULT]: is_valid={is_valid}, detected={detected}")
            return is_valid, detected
        except Exception as e:
            logger.error(f"Animal validation failed: {e}")
            print(f"[VALIDATION ERROR]: {e}")
            raise ValueError(f"VALIDATION_ERROR:{e}")

    def predict(self, image_content: bytes) -> Dict:
        """
        Full breed & age prediction (only called after validation passes).
        """
        if not self.enabled:
            raise ValueError("AI API Key not configured.")

        prompt = f"""This image contains a cow (cattle) or buffalo. Analyze it carefully.

Identify:
1. Animal Type: exactly "Cattle" or "Buffalo"
2. Breed: choose the single best match from this list: {', '.join(ALL_BREEDS)}
3. Age Group: choose from: {', '.join(AGE_GROUPS)}

Respond ONLY with this JSON (no extra text):
{{
    "animal_type": {{"prediction": "Cattle or Buffalo", "confidence": 0.95}},
    "breed": {{
        "prediction": "breed_name",
        "confidence": 0.90,
        "top_predictions": [
            {{"breed": "breed1", "confidence": 0.90}},
            {{"breed": "breed2", "confidence": 0.06}},
            {{"breed": "breed3", "confidence": 0.04}}
        ]
    }},
    "age": {{
        "prediction": "age_group",
        "confidence": 0.85,
        "all_predictions": [
            {{"age_group": "group1", "confidence": 0.85}},
            {{"age_group": "group2", "confidence": 0.10}},
            {{"age_group": "group3", "confidence": 0.05}}
        ]
    }}
}}
"""
        try:
            text = self._call_gemini(prompt, image_content)
            result = json.loads(_extract_json_object(text))

            animal_type = _normalize_animal_type(result.get("animal_type", {}).get("prediction", ""))
            breed_prediction = _normalize_breed(result.get("breed", {}).get("prediction", ""))
            age_prediction = _normalize_age_group(result.get("age", {}).get("prediction", ""))

            if "animal_type" in result and isinstance(result["animal_type"], dict):
                result["animal_type"]["prediction"] = animal_type
            if "breed" in result and isinstance(result["breed"], dict):
                result["breed"]["prediction"] = breed_prediction
                tops = result["breed"].get("top_predictions")
                if isinstance(tops, list):
                    result["breed"]["top_predictions"] = tops[:3]
            if "age" in result and isinstance(result["age"], dict):
                result["age"]["prediction"] = age_prediction
                all_preds = result["age"].get("all_predictions")
                if isinstance(all_preds, list):
                    for item in all_preds:
                        if isinstance(item, dict) and "age_group" in item:
                            item["age_group"] = _normalize_age_group(item["age_group"])
                    result["age"]["all_predictions"] = all_preds

            return result
        except Exception as e:
            logger.error(f"Error calling Gemini predict: {e}")
            raise

    def get_breed_info(self, breed_name: str) -> Dict:
        from .classifier import CattleClassifier

        temp_clf = CattleClassifier()
        return temp_clf.get_breed_info(breed_name)
