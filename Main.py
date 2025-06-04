import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from DataConfig import DataConfig
from ModelTrainer import ModelTrainer
from MultimodalDataLoader import MultimodalDataLoader


# Configure Chinese font support
def setup_chinese_font():
    """Configure matplotlib to support Chinese fonts"""
    chinese_fonts = [
        'PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS',
        'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei'
    ]
    plt.rcParams['font.sans-serif'] = chinese_fonts
    plt.rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')

setup_chinese_font()

# ======================== Main Program ========================
def main():
    """Main program"""
    print("🎯 Multimodal Lie Detection System")
    print("=" * 60)

    # 1. Configure and check data paths
    config = DataConfig()
    if not config.check_paths():
        print("❌ Data path check failed. Please ensure all data files exist.")
        return

    # 2. Load multimodal data
    data_loader = MultimodalDataLoader(config)
    data_result = data_loader.load_all_data()

    if data_result is None:
        print("❌ Data loading failed")
        return

    # Extract data from a dictionary
    audio_features = data_result.get('audio_features')
    text_features = data_result.get('text_features')
    visual_features = data_result.get('visual_features')
    annotation_features = data_result.get('annotation_features')

    # Extract labels - use the most complete label set
    audio_labels = data_result.get('audio_labels')
    text_labels = data_result.get('text_labels')
    visual_labels = data_result.get('visual_labels')
    annotation_labels = data_result.get('annotation_labels')

    # Select the largest label set as the primary label
    all_labels = []
    for label_set in [audio_labels, text_labels, visual_labels, annotation_labels]:
        if label_set is not None:
            all_labels.append(label_set)

    if not all_labels:
        print("❌ No valid labels found")
        return

    # Use the longest label set
    labels = max(all_labels, key=len)
    print(f"📊 Using label set with size: {len(labels)}")

    # Check if at least one modality has data
    available_modalities = []
    if audio_features is not None:
        available_modalities.append('audio')
    if text_features is not None:
        available_modalities.append('text')
    if visual_features is not None:
        available_modalities.append('visual')
    if annotation_features is not None:
        available_modalities.append('annotation')

    if not available_modalities:
        print("Failed to load features from any modality")
        return

    print(f"Successfully loaded modalities: {available_modalities}")

    # 3. Train models with different fusion strategies
    trainer = ModelTrainer()

    # Train concatenation fusion model
    print("\n" + "=" * 60)
    print("🤖 Training Concat Fusion Model...")
    try:
        concat_model, concat_scalers = trainer.train_model(
            audio_features, text_features, visual_features, annotation_features, labels,
            fusion_type='concat', epochs=80, batch_size=16
        )
        print("✅ Concat fusion model training completed")
    except Exception as e:
        print(f"❌ Concat fusion model training failed: {e}")
        concat_model, concat_scalers = None, None

    # Train attention fusion model
    print("\n" + "=" * 60)
    print("🤖 Training Attention Fusion Model...")
    try:
        attention_model, attention_scalers = trainer.train_model(
            audio_features, text_features, visual_features, annotation_features, labels,
            fusion_type='attention', epochs=80, batch_size=16
        )
        print("✅ Attention fusion model training completed")
    except Exception as e:
        print(f"❌ Attention fusion model training failed: {e}")
        attention_model, attention_scalers = None, None

    # Train weighted fusion model
    print("\n" + "=" * 60)
    print("🤖 Training Weighted Fusion Model...")
    try:
        weighted_model, weighted_scalers = trainer.train_model(
            audio_features, text_features, visual_features, annotation_features, labels,
            fusion_type='weighted', epochs=80, batch_size=16
        )
        print("✅ Weighted fusion model training completed")
    except Exception as e:
        print(f"❌ Weighted fusion model training failed: {e}")
        weighted_model, weighted_scalers = None, None

    # 4. Save trained models
    print("\n💾 Saving trained models...")
    saved_models = []

    if concat_model is not None:
        torch.save(concat_model.state_dict(), 'concat_fusion_model.pth')
        saved_models.append('concat')

    if attention_model is not None:
        torch.save(attention_model.state_dict(), 'attention_fusion_model.pth')
        saved_models.append('attention')

    if weighted_model is not None:
        torch.save(weighted_model.state_dict(), 'weighted_fusion_model.pth')
        saved_models.append('weighted')

    # Save feature scalers
    import pickle
    if concat_scalers is not None:
        with open('concat_scalers.pkl', 'wb') as f:
            pickle.dump(concat_scalers, f)
    if attention_scalers is not None:
        with open('attention_scalers.pkl', 'wb') as f:
            pickle.dump(attention_scalers, f)
    if weighted_scalers is not None:
        with open('weighted_scalers.pkl', 'wb') as f:
            pickle.dump(weighted_scalers, f)

    print(f"✅ Models saved successfully! Saved models: {saved_models}")

    # 5. Feature importance analysis
    print("\n📊 Generating feature analysis charts...")
    try:
        analyze_multimodal_features(audio_features, text_features, visual_features, annotation_features, labels)
        print("✅ Feature analysis completed")
    except Exception as e:
        print(f"❌ Feature analysis failed: {e}")
        import traceback
        traceback.print_exc()


def analyze_multimodal_features(audio_features, text_features, visual_features, annotation_features, labels):
    """Multimodal feature analysis"""

    # 1. Modality feature importance
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))

    # Audio feature importance
    audio_importance = np.var(audio_features, axis=0)
    top_audio_idx = np.argsort(audio_importance)[::-1][:15]

    axes[0, 0].bar(range(len(top_audio_idx)), audio_importance[top_audio_idx])
    axes[0, 0].set_title('Audio Feature Importance (Top 15)')
    axes[0, 0].set_xlabel('Feature Index')
    axes[0, 0].set_ylabel('Importance Score')

    # Text feature importance
    text_importance = np.var(text_features, axis=0)
    top_text_idx = np.argsort(text_importance)[::-1][:15]

    axes[0, 1].bar(range(len(top_text_idx)), text_importance[top_text_idx])
    axes[0, 1].set_title('Text Feature Importance (Top 15)')
    axes[0, 1].set_xlabel('Feature Index')
    axes[0, 1].set_ylabel('Importance Score')

    # Visual feature importance
    visual_importance = np.var(visual_features, axis=0)
    top_visual_idx = np.argsort(visual_importance)[::-1][:15]

    axes[1, 0].bar(range(len(top_visual_idx)), visual_importance[top_visual_idx])
    axes[1, 0].set_title('Visual Feature Importance (Top 15)')
    axes[1, 0].set_xlabel('Feature Index')
    axes[1, 0].set_ylabel('Importance Score')

    # Annotation feature importance
    annotation_importance = np.var(annotation_features, axis=0)
    top_annotation_idx = np.argsort(annotation_importance)[::-1][:15]

    axes[1, 1].bar(range(len(top_annotation_idx)), annotation_importance[top_annotation_idx])
    axes[1, 1].set_title('Annotation Feature Importance (Top 15)')
    axes[1, 1].set_xlabel('Feature Index')
    axes[1, 1].set_ylabel('Importance Score')

    plt.tight_layout()
    plt.show()

    # 2. Cross-modality correlation analysis
    plt.figure(figsize=(12, 8))

    # Compute average features per modality
    audio_mean = np.mean(audio_features, axis=1)
    text_mean = np.mean(text_features, axis=1)
    visual_mean = np.mean(visual_features, axis=1)
    annotation_mean = np.mean(annotation_features, axis=1)

    modality_data = np.column_stack([audio_mean, text_mean, visual_mean, annotation_mean])
    correlation_matrix = np.corrcoef(modality_data.T)

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=['Audio', 'Text', 'Visual', 'Annotation'],
                yticklabels=['Audio', 'Text', 'Visual', 'Annotation'])
    plt.title('Cross-Modality Correlation Analysis')
    plt.show()

    # 3. Class distribution visualization
    plt.figure(figsize=(15, 5))

    # Distribution of features under different classes for each modality
    modalities = ['Audio', 'Text', 'Visual', 'Annotation']
    features_list = [audio_features, text_features, visual_features, annotation_features]

    for i, (modality, features) in enumerate(zip(modalities, features_list)):
        plt.subplot(1, 4, i + 1)

        true_features = features[labels == 0]
        false_features = features[labels == 1]

        plt.hist(np.mean(true_features, axis=1), alpha=0.7, label='Truth', bins=20)
        plt.hist(np.mean(false_features, axis=1), alpha=0.7, label='Lie', bins=20)

        plt.title(f'{modality} Feature Distribution')
        plt.xlabel('Feature Value')
        plt.ylabel('Frequency')
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
