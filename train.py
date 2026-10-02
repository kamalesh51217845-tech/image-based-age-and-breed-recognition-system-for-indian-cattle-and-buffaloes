"""
Training Script for Indian Cattle & Buffalo Recognition Model
Run this script to train the model on your custom dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import json
from datetime import datetime
from tqdm import tqdm

from app.models.classifier import MultiTaskCattleNet, CattleDataset, ALL_BREEDS, AGE_GROUPS, ANIMAL_TYPES
from app.models.preprocessing import ImagePreprocessor


class Trainer:
    """Trainer class for the cattle classification model."""

    def __init__(self, data_dir: str, batch_size: int = 32, learning_rate: float = 0.001,
                 epochs: int = 50, device: str = None):
        """
        Initialize the trainer.

        Args:
            data_dir: Directory containing the training data
                      Expected structure:
                      data_dir/
                        images/
                          image1.jpg
                          image2.jpg
                          ...
                        labels.json  # {"image1.jpg": {"breed": "Gir", "age": "Adult", "type": "Cattle"}, ...}
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            device: Device to train on ('cuda' or 'cpu')
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize model
        self.model = MultiTaskCattleNet(pretrained=True)
        self.model.to(self.device)

        # Loss functions
        self.criterion_breed = nn.CrossEntropyLoss()
        self.criterion_age = nn.CrossEntropyLoss()
        self.criterion_type = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Preprocessor
        self.preprocessor = ImagePreprocessor(target_size=(224, 224))

        # Label mappings
        self.breed_to_idx = {breed: i for i, breed in enumerate(ALL_BREEDS)}
        self.age_to_idx = {age: i for i, age in enumerate(AGE_GROUPS)}
        self.type_to_idx = {t: i for i, t in enumerate(ANIMAL_TYPES)}

    def load_data(self):
        """Load and prepare the dataset."""
        images_dir = os.path.join(self.data_dir, 'images')
        labels_file = os.path.join(self.data_dir, 'labels.json')

        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

        with open(labels_file, 'r') as f:
            labels = json.load(f)

        image_paths = []
        breed_labels = []
        age_labels = []
        type_labels = []

        for filename, label_data in labels.items():
            image_path = os.path.join(images_dir, filename)
            if os.path.exists(image_path):
                image_paths.append(image_path)

                # Map labels to indices
                breed = label_data.get('breed', ALL_BREEDS[0])
                age = label_data.get('age', AGE_GROUPS[0])
                animal_type = label_data.get('type', ANIMAL_TYPES[0])

                breed_labels.append(self.breed_to_idx.get(breed, 0))
                age_labels.append(self.age_to_idx.get(age, 0))
                type_labels.append(self.type_to_idx.get(animal_type, 0))

        print(f"Loaded {len(image_paths)} images")

        # Create dataset
        dataset = CattleDataset(
            image_paths=image_paths,
            breed_labels=breed_labels,
            age_labels=age_labels,
            animal_type_labels=type_labels,
            preprocessor=self.preprocessor
        )

        # Split into train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

        print(f"Training samples: {train_size}, Validation samples: {val_size}")

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct_breed = 0
        correct_age = 0
        correct_type = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            images = batch['image'].to(self.device)
            breed_targets = batch['breed'].to(self.device)
            age_targets = batch['age'].to(self.device)
            type_targets = batch['animal_type'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            type_logits, breed_logits, age_logits = self.model(images)

            # Calculate losses
            loss_breed = self.criterion_breed(breed_logits, breed_targets)
            loss_age = self.criterion_age(age_logits, age_targets)
            loss_type = self.criterion_type(type_logits, type_targets)

            # Combined loss (weighted)
            loss = loss_breed + 0.5 * loss_age + 0.5 * loss_type

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, pred_breed = breed_logits.max(1)
            _, pred_age = age_logits.max(1)
            _, pred_type = type_logits.max(1)

            correct_breed += pred_breed.eq(breed_targets).sum().item()
            correct_age += pred_age.eq(age_targets).sum().item()
            correct_type += pred_type.eq(type_targets).sum().item()
            total += breed_targets.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'breed_acc': f'{100.*correct_breed/total:.2f}%'
            })

        return {
            'loss': total_loss / len(self.train_loader),
            'breed_acc': 100. * correct_breed / total,
            'age_acc': 100. * correct_age / total,
            'type_acc': 100. * correct_type / total
        }

    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct_breed = 0
        correct_age = 0
        correct_type = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                images = batch['image'].to(self.device)
                breed_targets = batch['breed'].to(self.device)
                age_targets = batch['age'].to(self.device)
                type_targets = batch['animal_type'].to(self.device)

                type_logits, breed_logits, age_logits = self.model(images)

                loss_breed = self.criterion_breed(breed_logits, breed_targets)
                loss_age = self.criterion_age(age_logits, age_targets)
                loss_type = self.criterion_type(type_logits, type_targets)
                loss = loss_breed + 0.5 * loss_age + 0.5 * loss_type

                total_loss += loss.item()
                _, pred_breed = breed_logits.max(1)
                _, pred_age = age_logits.max(1)
                _, pred_type = type_logits.max(1)

                correct_breed += pred_breed.eq(breed_targets).sum().item()
                correct_age += pred_age.eq(age_targets).sum().item()
                correct_type += pred_type.eq(type_targets).sum().item()
                total += breed_targets.size(0)

        return {
            'loss': total_loss / len(self.val_loader),
            'breed_acc': 100. * correct_breed / total,
            'age_acc': 100. * correct_age / total,
            'type_acc': 100. * correct_type / total
        }

    def train(self):
        """Run the full training loop."""
        print("\nStarting training...")
        print(f"Epochs: {self.epochs}, Batch size: {self.batch_size}, LR: {self.learning_rate}")
        print("-" * 60)

        best_val_loss = float('inf')
        history = []

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            # Update learning rate
            self.scheduler.step(val_metrics['loss'])

            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Breed Acc: {train_metrics['breed_acc']:.2f}%, "
                  f"Age Acc: {train_metrics['age_acc']:.2f}%")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Breed Acc: {val_metrics['breed_acc']:.2f}%, "
                  f"Age Acc: {val_metrics['age_acc']:.2f}%")

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_model('saved_models/cattle_classifier.pth')
                print("  → Saved best model!")

            history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics
            })

        # Save training history
        with open('saved_models/training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        print("\nTraining complete!")
        return history

    def save_model(self, path: str):
        """Save model weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)


def create_sample_labels():
    """Create a sample labels.json file for reference."""
    sample_labels = {
        "image1.jpg": {
            "breed": "Gir",
            "age": "Adult (18 months - 6 years)",
            "type": "Cattle"
        },
        "image2.jpg": {
            "breed": "Murrah",
            "age": "Young (6-18 months)",
            "type": "Buffalo"
        }
    }

    os.makedirs('data', exist_ok=True)
    with open('data/sample_labels.json', 'w') as f:
        json.dump(sample_labels, f, indent=2)

    print("Sample labels file created at: data/sample_labels.json")
    print("\nExpected data structure:")
    print("  data/")
    print("    images/")
    print("      image1.jpg")
    print("      image2.jpg")
    print("      ...")
    print("    labels.json")
    print("\nSupported breeds:", ALL_BREEDS)
    print("Age groups:", AGE_GROUPS)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train the cattle recognition model')
    parser.add_argument('--data', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--create-sample', action='store_true',
                        help='Create sample labels file')

    args = parser.parse_args()

    if args.create_sample:
        create_sample_labels()
    else:
        trainer = Trainer(
            data_dir=args.data,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr
        )
        trainer.load_data()
        trainer.train()
