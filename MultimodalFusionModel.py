import torch.nn as nn
import torch


# ======================== 多模态融合模型 ========================
class MultimodalFusionModel(nn.Module):
    """多模态融合模型"""

    def __init__(self, audio_dim, text_dim, visual_dim, annotation_dim,
                 hidden_dim=256, num_classes=2, fusion_type='attention'):
        super(MultimodalFusionModel, self).__init__()

        self.fusion_type = fusion_type

        # 各模态的特征投影层
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
            # 简单拼接融合
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
            # 注意力融合
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1)
            self.norm = nn.LayerNorm(hidden_dim)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, num_classes)
            )

        elif fusion_type == 'weighted':
            # 加权融合
            self.modality_weights = nn.Parameter(torch.ones(4))
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, num_classes)
            )

    def forward(self, audio_features, text_features, visual_features, annotation_features):
        # 特征投影
        audio_proj = self.audio_proj(audio_features)
        text_proj = self.text_proj(text_features)
        visual_proj = self.visual_proj(visual_features)
        annotation_proj = self.annotation_proj(annotation_features)

        if self.fusion_type == 'concat':
            # 拼接融合
            fused_features = torch.cat([audio_proj, text_proj, visual_proj, annotation_proj], dim=1)
            output = self.classifier(fused_features)

        elif self.fusion_type == 'attention':
            # 注意力融合
            # 形状: (seq_len=4, batch_size, hidden_dim)
            features_seq = torch.stack([audio_proj, text_proj, visual_proj, annotation_proj], dim=0)
            attended_features, attention_weights = self.attention(features_seq, features_seq, features_seq)
            # 平均池化
            fused_features = torch.mean(attended_features, dim=0)
            fused_features = self.norm(fused_features)
            output = self.classifier(fused_features)

        elif self.fusion_type == 'weighted':
            # 加权融合
            weights = torch.softmax(self.modality_weights, dim=0)
            fused_features = (weights[0] * audio_proj +
                              weights[1] * text_proj +
                              weights[2] * visual_proj +
                              weights[3] * annotation_proj)
            output = self.classifier(fused_features)

        return output
