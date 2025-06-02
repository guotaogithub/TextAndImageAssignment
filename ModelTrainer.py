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


# ======================== 训练和评估 ========================
class ModelTrainer:
    """模型训练器"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

    def train_model(self, audio_features, text_features, visual_features,
                    annotation_features, labels, fusion_type='attention',
                    epochs=100, batch_size=16):
        """训练模型"""
        print(f"\n🚀 开始训练 {fusion_type} 融合模型...")

        # 数据预处理
        audio_scaler = StandardScaler()
        text_scaler = StandardScaler()
        visual_scaler = StandardScaler()
        annotation_scaler = StandardScaler()

        audio_features = audio_scaler.fit_transform(audio_features)
        text_features = text_scaler.fit_transform(text_features)
        visual_features = visual_scaler.fit_transform(visual_features)
        annotation_features = annotation_scaler.fit_transform(annotation_features)

        # 数据分割
        indices = np.arange(len(labels))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=labels
        )

        # 转换为PyTorch张量
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

        # 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(
            X_train_audio, X_train_text, X_train_visual, X_train_annotation, y_train
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 创建模型
        model = MultimodalFusionModel(
            audio_dim=audio_features.shape[1],
            text_dim=text_features.shape[1],
            visual_dim=visual_features.shape[1],
            annotation_dim=annotation_features.shape[1],
            fusion_type=fusion_type
        ).to(self.device)

        # 训练设置
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

        # 训练历史
        train_losses = []
        train_accuracies = []

        # 训练循环
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

        # 测试评估
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

        # 生成评估报告
        self.generate_evaluation_report(y_test_np, test_predicted, fusion_type)

        # 绘制训练曲线
        self.plot_training_curves(train_losses, train_accuracies, fusion_type)

        return model, (audio_scaler, text_scaler, visual_scaler, annotation_scaler)

    def generate_evaluation_report(self, y_true, y_pred, model_name):
        """生成评估报告"""
        print(f"\n=== 📈 {model_name} 模型性能报告 ===")

        # 分类报告
        report = classification_report(y_true, y_pred, target_names=['真话', '假话'])
        print(report)

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['真话', '假话'], yticklabels=['真话', '假话'])
        plt.title(f'{model_name} 模型混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.show()

        # 关键指标
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        print(f"\n=== 🎯 关键性能指标 ===")
        print(f"准确率: {accuracy:.4f}")
        print(f"F1分数: {f1:.4f}")

    def plot_training_curves(self, losses, accuracies, model_name):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 损失曲线
        ax1.plot(losses, 'b-', label='Training Loss')
        ax1.set_title(f'{model_name} 模型训练损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # 准确率曲线
        ax2.plot(accuracies, 'r-', label='Training Accuracy')
        ax2.set_title(f'{model_name} 模型训练准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()
