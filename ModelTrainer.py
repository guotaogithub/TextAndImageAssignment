import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from MultimodalFusionModel import MultimodalFusionModel


# ======================== Training and Evaluation ========================
class ModelTrainer:
    """Model trainer class"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

    def train_model(self, audio_features, text_features, visual_features,
                    annotation_features, labels, fusion_type='attention',
                    epochs=100, batch_size=16):
        """Train the model"""
        print(f"\n🚀 Starting training of {fusion_type} fusion model...")

        # Data preprocessing
        audio_scaler = StandardScaler()
        text_scaler = StandardScaler()
        visual_scaler = StandardScaler()
        annotation_scaler = StandardScaler()

        audio_features = audio_scaler.fit_transform(audio_features)
        text_features = text_scaler.fit_transform(text_features)
        visual_features = visual_scaler.fit_transform(visual_features)
        annotation_features = annotation_scaler.fit_transform(annotation_features)

        # Data split
        indices = np.arange(len(labels))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=labels
        )

        # Convert to PyTorch tensors
        X_train_audio = torch.FloatTensor(audio_features[train_idx])
        X_train_text = torch.FloatTensor(text_features[train_idx])
        X_train_visual = torch.FloatTensor(visual_features[train_idx])
        X_train_annotation = torch.FloatTensor(annotation_features[train_idx])
        y_train = torch.LongTensor(labels[train_idx])

        X_test_audio = torch.FloatTensor(audio_features[test_idx])
        X_test_text = torch.FloatTensor(text_features[test_idx])
        X_test_visual = torch.FloatTensor(visual_features[test_idx])
        X_test_annotation = torch.FloatTensor(annotation_features[test_idx])
        y_test = torch.LongTensor(labels[test_idx])

        # Create data loader
        train_dataset = torch.utils.data.TensorDataset(
            X_train_audio, X_train_text, X_train_visual, X_train_annotation, y_train
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Build model
        model = MultimodalFusionModel(
            audio_dim=audio_features.shape[1],
            text_dim=text_features.shape[1],
            visual_dim=visual_features.shape[1],
            annotation_dim=annotation_features.shape[1],
            fusion_type=fusion_type
        ).to(self.device)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

        # Training history
        train_losses = []
        train_accuracies = []

        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch in train_loader:
                audio, text, visual, annotation, target = [x.to(self.device) for x in batch]

                optimizer.zero_grad()
                output = model(audio, text, visual, annotation)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            scheduler.step()

            avg_loss = total_loss / len(train_loader)
            accuracy = 100. * correct / total

            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)

            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Test evaluation
        model.eval()
        with torch.no_grad():
            X_test_audio = X_test_audio.to(self.device)
            X_test_text = X_test_text.to(self.device)
            X_test_visual = X_test_visual.to(self.device)
            X_test_annotation = X_test_annotation.to(self.device)

            test_output = model(X_test_audio, X_test_text, X_test_visual, X_test_annotation)
            _, test_predicted = torch.max(test_output.data, 1)
            test_predicted = test_predicted.cpu().numpy()
            y_test_np = y_test.numpy()

        # Generate evaluation report
        self.generate_evaluation_report(y_test_np, test_predicted, fusion_type)

        # Plot training curves
        self.plot_training_curves(train_losses, train_accuracies, fusion_type)

        return model, (audio_scaler, text_scaler, visual_scaler, annotation_scaler)

    def generate_evaluation_report(self, y_true, y_pred, model_name):
        """Generate evaluation report"""
        print(f"\n=== 📈 {model_name} Model Performance Report ===")

        # Classification report
        report = classification_report(y_true, y_pred, target_names=['Truth', 'Lie'])
        print(report)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Truth', 'Lie'], yticklabels=['Truth', 'Lie'])
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

        # Key metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        print(f"\n=== 🎯 Key Performance Metrics ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

    def plot_training_curves(self, losses, accuracies, model_name):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curve
        ax1.plot(losses, 'b-', label='Training Loss')
        ax1.set_title(f'{model_name} Training Loss Curve')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy curve
        ax2.plot(accuracies, 'r-', label='Training Accuracy')
        ax2.set_title(f'{model_name} Training Accuracy Curve')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
