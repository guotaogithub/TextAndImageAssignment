import torch.nn as nn
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score


class LieDetectionTrainer:
    """Trainer for a lie detection model"""

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_f1_scores = []
        self.val_f1_scores = []

    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in train_loader:
            audio_features = batch['audio'].to(self.device)
            text_features = batch['text'].to(self.device)
            visual_features = batch['visual'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(audio_features, text_features, visual_features)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Collect predictions
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate F1 score
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return total_loss / len(train_loader), f1

    def evaluate(self, val_loader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                audio_features = batch['audio'].to(self.device)
                text_features = batch['text'].to(self.device)
                visual_features = batch['visual'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(audio_features, text_features, visual_features)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average='weighted')

        return total_loss / len(val_loader), f1, all_preds, all_labels

    def train(self, train_loader, val_loader, epochs=50):
        """Full training process"""
        print("Start training multimodal lie detection model...")

        for epoch in range(epochs):
            # Train
            train_loss, train_f1 = self.train_epoch(train_loader)

            # Validate
            val_loss, val_f1, val_preds, val_labels = self.evaluate(val_loader)

            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_f1_scores.append(train_f1)
            self.val_f1_scores.append(val_f1)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}]')
                print(f'Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}')
                print(f'Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')
                print('-' * 50)

        return val_preds, val_labels

    def plot_training_history(self):
        """Plot training history curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curve
        ax1.plot(self.train_losses, label='Training Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training Loss Curve')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # F1 score curve
        ax2.plot(self.train_f1_scores, label='Training F1')
        ax2.plot(self.val_f1_scores, label='Validation F1')
        ax2.set_title('F1 Score Curve')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.legend()

        plt.tight_layout()
        plt.show()
