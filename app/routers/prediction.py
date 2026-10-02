"""
API Router for prediction endpoints.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional
import os
import uuid
from datetime import datetime
from io import BytesIO

from ..database.db import get_db, PredictionRepository, BreedRepository
from ..schemas.prediction import (
    PredictionResponse, PredictionHistoryItem, FeedbackRequest,
    BreedInfoCreate, BreedInfoResponse, StatisticsResponse, ErrorResponse
)
from ..config import settings
from ..models.preprocessing import ImagePreprocessor
from ..models.classifier import CattleClassifier, ALL_BREEDS, AGE_GROUPS
from ..models.external_ai import ExternalAIClassifier

router = APIRouter(prefix="/api", tags=["predictions"])

# Initialize preprocessor
preprocessor = ImagePreprocessor(target_size=(224, 224))
classifier = None  # Local classifier
external_ai = None  # AI API classifier

UPLOAD_DIR = settings.UPLOAD_DIR
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Lazy-loaded local animal validator (ImageNet)
_animal_validator_model = None
_animal_validator_transforms = None
_animal_validator_categories = None
_allowed_imagenet_labels = {"ox", "water buffalo", "bison", "yak"}


def _get_local_animal_validator():
    """
    Returns a tuple of (model, transforms, categories) for ImageNet-based validation.

    Note: this may download pretrained weights on first run (torchvision).
    """
    global _animal_validator_model, _animal_validator_transforms, _animal_validator_categories

    if _animal_validator_model is not None:
        return _animal_validator_model, _animal_validator_transforms, _animal_validator_categories

    try:
        import torch
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = mobilenet_v3_small(weights=weights)
        model.eval()
        transforms = weights.transforms()
        categories = weights.meta.get("categories") or []

        _animal_validator_model = model
        _animal_validator_transforms = transforms
        _animal_validator_categories = categories
        return _animal_validator_model, _animal_validator_transforms, _animal_validator_categories
    except Exception:
        # If torchvision/weights are unavailable for any reason, disable local validation.
        _animal_validator_model = False
        return None, None, None


def _local_validate_animal(image_bytes: bytes):
    """
    Local, offline-friendly-ish validator using an ImageNet pretrained model.

    Returns (is_valid, detected_label).
    - is_valid: True if top-k contains a bovine-like class label.
    - detected_label: top-1 ImageNet label when invalid, else best matching allowed label.
    """
    model, transforms, categories = _get_local_animal_validator()
    if model is None:
        return False, "unknown animal"

    try:
        import torch
        from PIL import Image

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        tensor = transforms(img).unsqueeze(0)

        with torch.inference_mode():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            top_probs, top_idxs = torch.topk(probs, k=5)

        top_idxs = top_idxs.tolist()
        top_probs = top_probs.tolist()

        top_labels = []
        for idx in top_idxs:
            if categories and 0 <= idx < len(categories):
                top_labels.append(categories[idx])
            else:
                top_labels.append(str(idx))

        # Accept if any allowed label is in top-k with some minimal signal
        best_allowed = None
        best_allowed_prob = 0.0
        for label, prob in zip(top_labels, top_probs):
            if label in _allowed_imagenet_labels and prob > best_allowed_prob:
                best_allowed = label
                best_allowed_prob = prob

        if best_allowed is not None and best_allowed_prob >= 0.03:
            return True, best_allowed

        return False, top_labels[0] if top_labels else "unknown animal"
    except Exception:
        return False, "unknown animal"


def get_classifier():
    """Lazy load the local classifier."""
    global classifier
    if classifier is None:
        classifier = CattleClassifier(model_path=settings.MODEL_PATH)
    return classifier


def get_external_ai():
    """Lazy load the external AI classifier if key is available."""
    global external_ai
    if external_ai is None:
        if settings.AI_API_KEY:
            candidate = ExternalAIClassifier(
                api_key=settings.AI_API_KEY,
                model_name=settings.AI_MODEL_NAME,
            )
            external_ai = candidate if getattr(candidate, "enabled", False) else False
        else:
            external_ai = False

    return None if external_ai is False else external_ai


def validate_image(file: UploadFile) -> bool:
    """Validate uploaded image file."""
    if not file.filename:
        return False
    ext = os.path.splitext(file.filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


@router.post("/predict", response_model=PredictionResponse)
async def predict_breed_and_age(
    file: UploadFile = File(...),
    save_to_db: bool = Form(default=True),
    db: Session = Depends(get_db)
):
    """
    Predict breed and age from uploaded cattle/buffalo image.

    - **file**: Image file (JPG, PNG, WEBP, BMP)
    - **save_to_db**: Whether to save prediction to database
    """
    # Validate file
    if not validate_image(file):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Allowed: JPG, PNG, WEBP, BMP"
        )

    try:
        # Read file content
        content = await file.read()

        ai_clf = get_external_ai()
        prediction = None
        breed_info = None

        # --- STEP 1: Animal validation (strict) ---
        # Prefer external validator when configured; otherwise fallback to local ImageNet validator.
        if ai_clf:
            try:
                print("Running animal validation via external AI...")
                is_valid, detected_animal = ai_clf.validate_animal(content)
            except Exception as e:
                print(f"External animal validation error (fallback to local): {e}")
                is_valid, detected_animal = _local_validate_animal(content)
        else:
            is_valid, detected_animal = _local_validate_animal(content)

        if not is_valid:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "animal_mismatch",
                    "message": f"Animal does not match. Detected: {detected_animal}. Only cow or buffalo images are accepted.",
                    "detected_animal": detected_animal,
                },
            )

        # Generate unique filename (only after validation passes)
        ext = os.path.splitext(file.filename)[1].lower()
        unique_filename = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)

        # Ensure upload directory exists
        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Save file
        with open(file_path, "wb") as f:
            f.write(content)

        # --- STEP 2: Full prediction ---
        if ai_clf:
            try:
                print(f"Using External AI ({settings.AI_PROVIDER}) for prediction...")
                prediction = ai_clf.predict(content)
                breed_info = ai_clf.get_breed_info(prediction["breed"]["prediction"])
            except Exception as e:
                print(f"External AI prediction failed: {e}. Falling back to local model.")
                prediction = None

        # Fallback to local model (only reached if AI prediction failed, animal already validated)
        if prediction is None:
            model_input, _ = preprocessor.preprocess(content)
            clf = get_classifier()
            prediction = clf.predict(model_input)
            breed_info = clf.get_breed_info(prediction["breed"]["prediction"])

        # Save to database if requested
        prediction_id = None
        if save_to_db:
            db_prediction = PredictionRepository.create_prediction(db, {
                "image_path": file_path,
                "original_filename": file.filename,
                "animal_type": prediction["animal_type"]["prediction"],
                "animal_type_confidence": prediction["animal_type"]["confidence"],
                "predicted_breed": prediction["breed"]["prediction"],
                "breed_confidence": prediction["breed"]["confidence"],
                "predicted_age": prediction["age"]["prediction"],
                "age_confidence": prediction["age"]["confidence"],
                "top_breeds": prediction["breed"]["top_predictions"]
            })
            prediction_id = db_prediction.id

        return PredictionResponse(
            success=True,
            prediction_id=prediction_id,
            image_path=f"/uploads/{unique_filename}",
            animal_type=prediction["animal_type"],
            breed=prediction["breed"],
            age=prediction["age"],
            breed_info=breed_info,
            message="Prediction successful"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions", response_model=list[PredictionHistoryItem])
async def get_prediction_history(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get prediction history with pagination."""
    predictions = PredictionRepository.get_all_predictions(db, skip=skip, limit=limit)
    return predictions


@router.delete("/predictions")
async def clear_prediction_history(
    delete_files: bool = True,
    db: Session = Depends(get_db)
):
    """Clear all prediction history (System Logs)."""
    result = PredictionRepository.clear_all_predictions(db, delete_files=delete_files)
    return {"success": True, **result}


@router.get("/predictions/{prediction_id}", response_model=PredictionHistoryItem)
async def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    """Get a specific prediction by ID."""
    prediction = PredictionRepository.get_prediction_by_id(db, prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction


@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest, db: Session = Depends(get_db)):
    """Submit feedback for a prediction (for model improvement)."""
    prediction = PredictionRepository.update_feedback(
        db,
        feedback.prediction_id,
        feedback.feedback,
        feedback.actual_breed,
        feedback.actual_age
    )
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {"success": True, "message": "Feedback submitted successfully"}


@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(db: Session = Depends(get_db)):
    """Get prediction statistics."""
    stats = PredictionRepository.get_statistics(db)
    return stats


@router.get("/breeds")
async def get_all_breeds():
    """Get list of all supported breeds."""
    clf = get_classifier()
    return {
        "cattle_breeds": clf.cattle_breeds,
        "buffalo_breeds": clf.buffalo_breeds,
        "all_breeds": clf.breeds
    }


@router.get("/age-groups")
async def get_age_groups():
    """Get list of age groups."""
    return {"age_groups": AGE_GROUPS}


# Admin endpoints for breed management
@router.post("/admin/breeds", response_model=BreedInfoResponse)
async def create_breed(breed: BreedInfoCreate, db: Session = Depends(get_db)):
    """Create a new breed entry (Admin)."""
    existing = BreedRepository.get_breed_by_name(db, breed.name)
    if existing:
        raise HTTPException(status_code=400, detail="Breed already exists")
    return BreedRepository.create_breed(db, breed.model_dump())


@router.get("/admin/breeds", response_model=list[BreedInfoResponse])
async def get_breed_info_list(db: Session = Depends(get_db)):
    """Get all breed information entries (Admin)."""
    return BreedRepository.get_all_breeds(db)


@router.put("/admin/breeds/{breed_id}", response_model=BreedInfoResponse)
async def update_breed(
    breed_id: int,
    breed: BreedInfoCreate,
    db: Session = Depends(get_db)
):
    """Update breed information (Admin)."""
    updated = BreedRepository.update_breed(db, breed_id, breed.model_dump())
    if not updated:
        raise HTTPException(status_code=404, detail="Breed not found")
    return updated


@router.delete("/admin/breeds/{breed_id}")
async def delete_breed(breed_id: int, db: Session = Depends(get_db)):
    """Delete a breed entry (Admin)."""
    if BreedRepository.delete_breed(db, breed_id):
        return {"success": True, "message": "Breed deleted"}
    raise HTTPException(status_code=404, detail="Breed not found")
