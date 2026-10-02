"""
CNN-based Classifier for Indian Cattle Breed and Age Recognition
Uses transfer learning with EfficientNet/MobileNet backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import numpy as np
import os


# Define breed and age categories
CATTLE_BREEDS = [
    "Gir",
    "Sahiwal",
    "Red Sindhi",
    "Ongole",
    "Tharparkar",
    "Kankrej",
    "Hariana",
    "Rathi",
    "Krishna Valley",
    "Deoni"
]

BUFFALO_BREEDS = [
    "Murrah",
    "Jaffarabadi",
    "Surti",
    "Mehsana",
    "Bhadawari",
    "Nili-Ravi",
    "Pandharpuri",
    "Nagpuri",
    "Toda"
]

ALL_BREEDS = CATTLE_BREEDS + BUFFALO_BREEDS

AGE_GROUPS = [
    "Calf (0-6 months)",
    "Young (6-18 months)",
    "Adult (18 months - 6 years)",
    "Old (6+ years)"
]

ANIMAL_TYPES = ["Cattle", "Buffalo"]


class MultiTaskCattleNet(nn.Module):
    """
    Multi-task CNN for simultaneous breed classification and age estimation.
    Uses a shared backbone with separate heads for each task.
    """

    def __init__(self, num_breeds: int = len(ALL_BREEDS),
                 num_age_groups: int = len(AGE_GROUPS),
                 num_animal_types: int = len(ANIMAL_TYPES),
                 pretrained: bool = True):
        super(MultiTaskCattleNet, self).__init__()

        # Use MobileNetV2 as backbone (lightweight and efficient)
        try:
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
            if pretrained:
                self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            else:
                self.backbone = mobilenet_v2(weights=None)
        except:
            from torchvision.models import mobilenet_v2
            self.backbone = mobilenet_v2(pretrained=pretrained)

        # Get the number of features from backbone
        num_features = self.backbone.classifier[1].in_features

        # Remove the original classifier
        self.backbone.classifier = nn.Identity()

        # Shared feature extractor after backbone
        self.shared_fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Animal type classification head (Cattle vs Buffalo)
        self.animal_type_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_animal_types)
        )

        # Breed classification head
        self.breed_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_breeds)
        )

        # Age estimation head
        self.age_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_age_groups)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            Tuple of (animal_type_logits, breed_logits, age_logits)
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Pass through shared layers
        shared_features = self.shared_fc(features)

        # Get predictions from each head
        animal_type_logits = self.animal_type_head(shared_features)
        breed_logits = self.breed_head(shared_features)
        age_logits = self.age_head(shared_features)

        return animal_type_logits, breed_logits, age_logits


class CattleClassifier:
    """
    High-level classifier interface for cattle breed and age prediction.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = None):
        """
        Initialize the classifier.

        Args:
            model_path: Path to saved model weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = MultiTaskCattleNet(pretrained=True)
        self.model.to(self.device)

        # Load saved weights if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("No trained model found. Using pretrained backbone with random heads.")
            print("For accurate predictions, train the model on cattle/buffalo dataset.")

        self.model.eval()

        # Store class labels
        self.breeds = ALL_BREEDS
        self.age_groups = AGE_GROUPS
        self.animal_types = ANIMAL_TYPES
        self.cattle_breeds = CATTLE_BREEDS
        self.buffalo_breeds = BUFFALO_BREEDS

    def load_model(self, model_path: str):
        """Load model weights from file."""
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def save_model(self, model_path: str):
        """Save model weights to file."""
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def predict(self, preprocessed_image: np.ndarray) -> Dict:
        """
        Make predictions on a preprocessed image.

        Args:
            preprocessed_image: Preprocessed image array of shape (1, 3, 224, 224)

        Returns:
            Dictionary containing predictions with confidence scores
        """
        # Convert to tensor
        input_tensor = torch.from_numpy(preprocessed_image).float().to(self.device)

        with torch.no_grad():
            animal_type_logits, breed_logits, age_logits = self.model(input_tensor)

            # Get probabilities
            animal_type_probs = F.softmax(animal_type_logits, dim=1).cpu().numpy()[0]
            breed_probs = F.softmax(breed_logits, dim=1).cpu().numpy()[0]
            age_probs = F.softmax(age_logits, dim=1).cpu().numpy()[0]

        # Get top predictions
        animal_type_idx = np.argmax(animal_type_probs)
        animal_type = self.animal_types[animal_type_idx]
        animal_type_confidence = float(animal_type_probs[animal_type_idx])

        # Filter breeds based on animal type
        if animal_type == "Cattle":
            relevant_breeds = self.cattle_breeds
            breed_indices = list(range(len(self.cattle_breeds)))
        else:
            relevant_breeds = self.buffalo_breeds
            breed_indices = list(range(len(self.cattle_breeds), len(self.breeds)))

        # Get breed prediction from relevant subset
        relevant_probs = breed_probs[breed_indices]
        breed_idx_in_subset = np.argmax(relevant_probs)
        predicted_breed = relevant_breeds[breed_idx_in_subset]
        breed_confidence = float(relevant_probs[breed_idx_in_subset])

        # Get age prediction
        age_idx = np.argmax(age_probs)
        predicted_age = self.age_groups[age_idx]
        age_confidence = float(age_probs[age_idx])

        # Get top 3 breed predictions
        top_breed_indices = np.argsort(relevant_probs)[::-1][:3]
        top_breeds = [
            {
                "breed": relevant_breeds[i],
                "confidence": float(relevant_probs[i])
            }
            for i in top_breed_indices
        ]

        # Get all age predictions
        age_predictions = [
            {
                "age_group": self.age_groups[i],
                "confidence": float(age_probs[i])
            }
            for i in range(len(self.age_groups))
        ]
        age_predictions.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "animal_type": {
                "prediction": animal_type,
                "confidence": animal_type_confidence
            },
            "breed": {
                "prediction": predicted_breed,
                "confidence": breed_confidence,
                "top_predictions": top_breeds
            },
            "age": {
                "prediction": predicted_age,
                "confidence": age_confidence,
                "all_predictions": age_predictions
            }
        }

    def get_breed_info(self, breed_name: str) -> Dict:
        """Get information about a specific breed."""
        breed_info = {
            # Cattle Breeds
            "Gir": {
                "origin": "Gujarat (Saurashtra)",
                "type": "Dairy",
                "characteristics": "Distinctive curved horns, drooping ears, and high forehead.",
                "average_milk_yield": "6-10 liters/day"
            },
            "Sahiwal": {
                "origin": "Punjab/Haryana",
                "type": "Dairy",
                "characteristics": "Heat tolerant, tick resistant, reddish-brown color, and docile temperament.",
                "average_milk_yield": "8-12 liters/day"
            },
            "Red Sindhi": {
                "origin": "Sindh region",
                "type": "Dairy",
                "characteristics": "Deep dark red color, compact body, and high adaptability to heat.",
                "average_milk_yield": "5-8 liters/day"
            },
            "Ongole": {
                "origin": "Andhra Pradesh (Nellore)",
                "type": "Dual purpose (Draft & Dairy)",
                "characteristics": "Large muscular build, white/grey coat, known for strength and endurance.",
                "average_milk_yield": "3-5 liters/day"
            },
            "Tharparkar": {
                "origin": "Rajasthan (Thar Desert)",
                "type": "Dual purpose",
                "characteristics": "White/grey coat, lyre-shaped horns, excellently adapted to desert conditions.",
                "average_milk_yield": "4-6 liters/day"
            },
            "Kankrej": {
                "origin": "Gujarat (Runn of Kutch)",
                "type": "Dual purpose",
                "characteristics": "Large silver-grey body, powerful lyre-shaped horns, 'Sawai Chal' gait.",
                "average_milk_yield": "5-8 liters/day"
            },
            "Hariana": {
                "origin": "Haryana (Rohtak, Hisar)",
                "type": "Dual purpose",
                "characteristics": "White or light grey color, small horns, excellent draft animal.",
                "average_milk_yield": "3-5 liters/day"
            },
            "Rathi": {
                "origin": "Rajasthan (Bikaner, Ganganagar)",
                "type": "Dairy",
                "characteristics": "Brown with white patches, medium build, efficient milk producer in arid zones.",
                "average_milk_yield": "5-7 liters/day"
            },
            "Krishna Valley": {
                "origin": "Karnataka (Krishna River basin)",
                "type": "Draft",
                "characteristics": "Grey-white color, massive body, used for heavy plowing.",
                "average_milk_yield": "2-3 liters/day"
            },
            "Deoni": {
                "origin": "Maharashtra (Latur, Osmanabad)",
                "type": "Dual purpose",
                "characteristics": "Black and white spotted coat, prominent forehead, drooping ears.",
                "average_milk_yield": "3-4 liters/day"
            },
            
            # Buffalo Breeds
            "Murrah": {
                "origin": "Haryana (Rohtak, Jind)",
                "type": "Dairy Buffalo",
                "characteristics": "Jet black, tightly curled 'corkscrew' horns, world's best dairy buffalo.",
                "average_milk_yield": "10-16 liters/day"
            },
            "Jaffarabadi": {
                "origin": "Gujarat (Saurashtra)",
                "type": "Dairy Buffalo",
                "characteristics": "Massive build, heavy drooping horns, highest fat content in milk.",
                "average_milk_yield": "8-12 liters/day"
            },
            "Surti": {
                "origin": "Gujarat (Kaira, Baroda)",
                "type": "Dairy Buffalo",
                "characteristics": "Medium sized, sickle-shaped horns, rusty brown to silver-grey color.",
                "average_milk_yield": "6-10 liters/day"
            },
            "Mehsana": {
                "origin": "Gujarat (Mehsana)",
                "type": "Dairy Buffalo",
                "characteristics": "Cross between Murrah and Surti, stable milk producer.",
                "average_milk_yield": "7-12 liters/day"
            },
            "Bhadawari": {
                "origin": "Uttar Pradesh (Agra, Etawah)",
                "type": "Dairy Buffalo",
                "characteristics": "Copper-colored coat, known for very high milk fat content (up to 13%).",
                "average_milk_yield": "3-5 liters/day"
            },
            "Nili-Ravi": {
                "origin": "Punjab (Sutlej and Ravi rivers)",
                "type": "Dairy Buffalo",
                "characteristics": "Wall eyes (white markings), black color, small curled horns.",
                "average_milk_yield": "6-10 liters/day"
            },
            "Pandharpuri": {
                "origin": "Maharashtra (Solapur, Kolhapur)",
                "type": "Dairy Buffalo",
                "characteristics": "Long sword-shaped horns (up to 50cm), medium build, fast reproductive cycle.",
                "average_milk_yield": "4-6 liters/day"
            },
            "Nagpuri": {
                "origin": "Maharashtra (Vidarbha)",
                "type": "Draft & Dairy Buffalo",
                "characteristics": "Long flat horns, black color, adapted to hot climates.",
                "average_milk_yield": "3-5 liters/day"
            },
            "Toda": {
                "origin": "Tamil Nadu (Nilgiris)",
                "type": "Semi-wild/Draft Buffalo",
                "characteristics": "Distinctive crescent-shaped horns, gregarious nature, thick hairy coat.",
                "average_milk_yield": "2-4 liters/day"
            }
        }
        return breed_info.get(breed_name, {"info": "Information not available"})


# For training purposes (can be expanded)
class CattleDataset(torch.utils.data.Dataset):
    """Dataset class for training the cattle classifier."""

    def __init__(self, image_paths: List[str], breed_labels: List[int],
                 age_labels: List[int], animal_type_labels: List[int],
                 preprocessor=None):
        self.image_paths = image_paths
        self.breed_labels = breed_labels
        self.age_labels = age_labels
        self.animal_type_labels = animal_type_labels
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        if self.preprocessor:
            image, _ = self.preprocessor.preprocess(image_path)
            image = torch.from_numpy(image).float().squeeze(0)
        else:
            from PIL import Image
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            image = Image.open(image_path).convert('RGB')
            image = transform(image)

        return {
            'image': image,
            'breed': self.breed_labels[idx],
            'age': self.age_labels[idx],
            'animal_type': self.animal_type_labels[idx]
        }
