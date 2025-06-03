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
        # Feature projection
        audio_proj = self.audio_proj(audio_features)
        text_proj = self.text_proj(text_features)
        visual_proj = self.visual_proj(visual_features)
        annotation_proj = self.annotation_proj(annotation_features)

        if self.fusion_type == 'concat':
            # Concatenation fusion
            fused_features = torch.cat([audio_proj, text_proj, visual_proj, annotation_proj], dim=1)
            output = self.classifier(fused_features)

        elif self.fusion_type == 'attention':
            # Attention-based fusion
            # Shape: (seq_len=4, batch_size, hidden_dim)
            features_seq = torch.stack([audio_proj, text_proj, visual_proj, annotation_proj], dim=0)
            attended_features, attention_weights = self.attention(features_seq, features_seq, features_seq)
            # Average pooling
            fused_features = torch.mean(attended_features, dim=0)
            fused_features = self.norm(fused_features)
            output = self.classifier(fused_features)

        elif self.fusion_type == 'weighted':
            # Weighted fusion
            weights = torch.softmax(self.modality_weights, dim=0)
            fused_features = (weights[0] * audio_proj +
                              weights[1] * text_proj +
                              weights[2] * visual_proj +
                              weights[3] * annotation_proj)
            output = self.classifier(fused_features)

        return output
