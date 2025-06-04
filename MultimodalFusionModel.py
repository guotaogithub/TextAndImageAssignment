import torch.nn as nn
import torch


# ======================== Multimodal Fusion Model ========================
class MultimodalFusionModel(nn.Module):
    """Multimodal fusion model"""

    def __init__(self, audio_dim, text_dim, visual_dim, annotation_dim,
                 hidden_dim=256, num_classes=2, fusion_type='attention'):
        super(MultimodalFusionModel, self).__init__()

        self.fusion_type = fusion_type

        # Feature projection layers for each modality
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.annotation_proj = nn.Sequential(
            nn.Linear(annotation_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        if fusion_type == 'concat':
            # Simple concatenation fusion
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes)
            )

        elif fusion_type == 'attention':
            # Attention-based fusion
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1)
            self.norm = nn.LayerNorm(hidden_dim)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, num_classes)
            )

        elif fusion_type == 'weighted':
            # Weighted fusion
            self.modality_weights = nn.Parameter(torch.ones(4))
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, num_classes)
            )

    def forward(self, audio_features, text_features, visual_features, annotation_features):
        projections = []

        if audio_features is not None and not (audio_features == 0).all():
            projections.append(self.audio_proj(audio_features))
        if text_features is not None and not (text_features == 0).all():
            projections.append(self.text_proj(text_features))
        if visual_features is not None and not (visual_features == 0).all():
            projections.append(self.visual_proj(visual_features))
        if annotation_features is not None and not (annotation_features == 0).all():
            projections.append(self.annotation_proj(annotation_features))

        if not projections:
            raise ValueError("No valid features provided for any modality")

        # 融合方式
        if self.fusion_type == 'concat':
            fused_features = torch.cat(projections, dim=1)
            output = self.classifier(fused_features)

        elif self.fusion_type == 'attention':
            features_seq = torch.stack(projections, dim=0)
            attended_features, _ = self.attention(features_seq, features_seq, features_seq)
            fused_features = torch.mean(attended_features, dim=0)
            fused_features = self.norm(fused_features)
            output = self.classifier(fused_features)

        elif self.fusion_type == 'weighted':
            weights = torch.softmax(self.modality_weights[:len(projections)], dim=0)
            weighted_sum = sum(w * f for w, f in zip(weights, projections))
            output = self.classifier(weighted_sum)

        return output  # 只返回主任务输出
