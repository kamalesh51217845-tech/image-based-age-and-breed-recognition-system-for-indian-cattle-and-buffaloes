"""
Database module for storing predictions and user uploads.
Uses SQLite with SQLAlchemy for simplicity.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Database configuration
DATABASE_URL = "sqlite:///./cattle_recognition.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Prediction(Base):
    """Model to store prediction history."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String(500), nullable=False)
    original_filename = Column(String(255))

    # Animal type prediction
    animal_type = Column(String(50))
    animal_type_confidence = Column(Float)

    # Breed prediction
    predicted_breed = Column(String(100))
    breed_confidence = Column(Float)

    # Age prediction
    predicted_age = Column(String(100))
    age_confidence = Column(Float)

    # Additional info
    top_breeds_json = Column(Text)  # JSON string of top breed predictions

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    user_feedback = Column(String(50))  # Optional: correct/incorrect feedback
    actual_breed = Column(String(100))  # Optional: user-provided correct breed
    actual_age = Column(String(100))  # Optional: user-provided correct age


class BreedInfo(Base):
    """Model to store breed information (for admin panel)."""
    __tablename__ = "breeds"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    animal_type = Column(String(50))  # Cattle or Buffalo
    origin = Column(String(200))
    breed_type = Column(String(100))  # Dairy, Draft, Dual purpose
    characteristics = Column(Text)
    average_milk_yield = Column(String(100))
    image_url = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Database operations
class PredictionRepository:
    """Repository class for prediction operations."""

    @staticmethod
    def create_prediction(db, prediction_data: dict) -> Prediction:
        """Create a new prediction record."""
        import json

        prediction = Prediction(
            image_path=prediction_data.get("image_path"),
            original_filename=prediction_data.get("original_filename"),
            animal_type=prediction_data.get("animal_type"),
            animal_type_confidence=prediction_data.get("animal_type_confidence"),
            predicted_breed=prediction_data.get("predicted_breed"),
            breed_confidence=prediction_data.get("breed_confidence"),
            predicted_age=prediction_data.get("predicted_age"),
            age_confidence=prediction_data.get("age_confidence"),
            top_breeds_json=json.dumps(prediction_data.get("top_breeds", []))
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        return prediction

    @staticmethod
    def get_all_predictions(db, skip: int = 0, limit: int = 100):
        """Get all predictions with pagination."""
        return db.query(Prediction).order_by(
            Prediction.created_at.desc()
        ).offset(skip).limit(limit).all()

    @staticmethod
    def get_prediction_by_id(db, prediction_id: int):
        """Get a specific prediction by ID."""
        return db.query(Prediction).filter(Prediction.id == prediction_id).first()

    @staticmethod
    def update_feedback(db, prediction_id: int, feedback: str,
                       actual_breed: str = None, actual_age: str = None):
        """Update user feedback for a prediction."""
        prediction = db.query(Prediction).filter(Prediction.id == prediction_id).first()
        if prediction:
            prediction.user_feedback = feedback
            if actual_breed:
                prediction.actual_breed = actual_breed
            if actual_age:
                prediction.actual_age = actual_age
            db.commit()
            db.refresh(prediction)
        return prediction

    @staticmethod
    def get_statistics(db):
        """Get overall prediction statistics."""
        from sqlalchemy import func

        total = db.query(func.count(Prediction.id)).scalar()

        breed_stats = db.query(
            Prediction.predicted_breed,
            func.count(Prediction.id).label('count')
        ).group_by(Prediction.predicted_breed).all()

        animal_type_stats = db.query(
            Prediction.animal_type,
            func.count(Prediction.id).label('count')
        ).group_by(Prediction.animal_type).all()

        return {
            "total_predictions": total,
            "breed_distribution": {b: c for b, c in breed_stats},
            "animal_type_distribution": {a: c for a, c in animal_type_stats}
        }

    @staticmethod
    def clear_all_predictions(db, delete_files: bool = True):
        """
        Delete all prediction history records.

        If delete_files is True, also attempts to delete the uploaded image files referenced by image_path.
        """
        predictions = db.query(Prediction).all()
        deleted_files = 0
        file_errors = 0

        if delete_files:
            for prediction in predictions:
                path = prediction.image_path
                if not path:
                    continue
                try:
                    if os.path.exists(path):
                        os.remove(path)
                        deleted_files += 1
                except Exception:
                    file_errors += 1

        deleted_records = len(predictions)
        if deleted_records > 0:
            for prediction in predictions:
                db.delete(prediction)
            db.commit()

        return {
            "deleted_records": deleted_records,
            "deleted_files": deleted_files,
            "file_errors": file_errors
        }


class BreedRepository:
    """Repository class for breed information operations."""

    @staticmethod
    def create_breed(db, breed_data: dict) -> BreedInfo:
        """Create a new breed entry."""
        breed = BreedInfo(**breed_data)
        db.add(breed)
        db.commit()
        db.refresh(breed)
        return breed

    @staticmethod
    def get_all_breeds(db):
        """Get all breeds."""
        return db.query(BreedInfo).all()

    @staticmethod
    def get_breed_by_name(db, name: str):
        """Get breed by name."""
        return db.query(BreedInfo).filter(BreedInfo.name == name).first()

    @staticmethod
    def update_breed(db, breed_id: int, breed_data: dict):
        """Update breed information."""
        breed = db.query(BreedInfo).filter(BreedInfo.id == breed_id).first()
        if breed:
            for key, value in breed_data.items():
                setattr(breed, key, value)
            db.commit()
            db.refresh(breed)
        return breed

    @staticmethod
    def delete_breed(db, breed_id: int):
        """Delete a breed."""
        breed = db.query(BreedInfo).filter(BreedInfo.id == breed_id).first()
        if breed:
            db.delete(breed)
            db.commit()
            return True
        return False


# Initialize database on import
create_tables()
