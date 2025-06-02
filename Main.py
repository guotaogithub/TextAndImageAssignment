import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from DataConfig import DataConfig
from ModelTrainer import ModelTrainer
from MultimodalDataLoader import MultimodalDataLoader


# 配置中文字体支持
def setup_chinese_font():
    """配置matplotlib中文字体支持"""
    chinese_fonts = [
        'PingFang SC', 'Hiragino Sans GB', 'Arial Unicode MS',
        'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei'
    ]
    plt.rcParams['font.sans-serif'] = chinese_fonts
    plt.rcParams['axes.unicode_minus'] = False
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')

setup_chinese_font()



# ======================== 主程序 ========================
def main():
    """主程序"""
    print("🎯 多模态谎言检测系统")
    print("=" * 60)

    # 1. 配置和检查数据路径
    config = DataConfig()
    if not config.check_paths():
        print("❌ 数据路径检查失败，请确保所有数据文件存在")
        return

    # 2. 加载多模态数据
    data_loader = MultimodalDataLoader(config)
    data_result = data_loader.load_all_data()

    if data_result is None:
        print("❌ 数据加载失败")
        return

    # 从字典中提取数据
    audio_features = data_result.get('audio_features')
    text_features = data_result.get('text_features')
    visual_features = data_result.get('visual_features')
    annotation_features = data_result.get('annotation_features')

    # 提取标签 - 使用最完整的标签集
    audio_labels = data_result.get('audio_labels')
    text_labels = data_result.get('text_labels')
    visual_labels = data_result.get('visual_labels')
    annotation_labels = data_result.get('annotation_labels')

    # 选择最大的标签集作为主要标签
    all_labels = []
    for label_set in [audio_labels, text_labels, visual_labels, annotation_labels]:
        if label_set is not None:
            all_labels.append(label_set)

    if not all_labels:
        print("❌ 没有找到有效的标签")
        return

    # 使用最长的标签集
    labels = max(all_labels, key=len)
    print(f"📊 使用标签集，大小: {len(labels)}")

    # 检查是否有至少一种模态的数据
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
        print("❌ 没有成功加载任何模态的特征")
        return

    print(f"✅ 成功加载模态: {available_modalities}")

    # 3. 训练不同融合策略的模型
    trainer = ModelTrainer()

    # 训练拼接融合模型
    print("\n" + "=" * 60)
    print("🤖 开始训练拼接融合模型...")
    try:
        concat_model, concat_scalers = trainer.train_model(
            audio_features, text_features, visual_features, annotation_features, labels,
            fusion_type='concat', epochs=80, batch_size=16
        )
        print("✅ 拼接融合模型训练完成")
    except Exception as e:
        print(f"❌ 拼接融合模型训练失败: {e}")
        concat_model, concat_scalers = None, None

    # 训练注意力融合模型
    print("\n" + "=" * 60)
    print("🤖 开始训练注意力融合模型...")
    try:
        attention_model, attention_scalers = trainer.train_model(
            audio_features, text_features, visual_features, annotation_features, labels,
            fusion_type='attention', epochs=80, batch_size=16
        )
        print("✅ 注意力融合模型训练完成")
    except Exception as e:
        print(f"❌ 注意力融合模型训练失败: {e}")
        attention_model, attention_scalers = None, None

    # 训练加权融合模型
    print("\n" + "=" * 60)
    print("🤖 开始训练加权融合模型...")
    try:
        weighted_model, weighted_scalers = trainer.train_model(
            audio_features, text_features, visual_features, annotation_features, labels,
            fusion_type='weighted', epochs=80, batch_size=16
        )
        print("✅ 加权融合模型训练完成")
    except Exception as e:
        print(f"❌ 加权融合模型训练失败: {e}")
        weighted_model, weighted_scalers = None, None

    # 4. 保存模型
    print("\n💾 保存训练好的模型...")
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

    # 保存特征缩放器
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

    print(f"✅ 模型保存完成！成功保存的模型: {saved_models}")

    # 5. 特征重要性分析
    print("\n📊 生成特征分析图表...")
    try:
        analyze_multimodal_features(audio_features, text_features, visual_features, annotation_features, labels)
        print("✅ 特征分析完成")
    except Exception as e:
        print(f"❌ 特征分析失败: {e}")
        import traceback
        traceback.print_exc()


def analyze_multimodal_features(audio_features, text_features, visual_features, annotation_features, labels):
    """多模态特征分析"""

    # 1. 各模态特征重要性
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))

    # 音频特征重要性
    audio_importance = np.var(audio_features, axis=0)
    top_audio_idx = np.argsort(audio_importance)[::-1][:15]

    axes[0, 0].bar(range(len(top_audio_idx)), audio_importance[top_audio_idx])
    axes[0, 0].set_title('音频特征重要性 (Top 15)')
    axes[0, 0].set_xlabel('特征索引')
    axes[0, 0].set_ylabel('重要性分数')

    # 文本特征重要性
    text_importance = np.var(text_features, axis=0)
    top_text_idx = np.argsort(text_importance)[::-1][:15]

    axes[0, 1].bar(range(len(top_text_idx)), text_importance[top_text_idx])
    axes[0, 1].set_title('文本特征重要性 (Top 15)')
    axes[0, 1].set_xlabel('特征索引')
    axes[0, 1].set_ylabel('重要性分数')

    # 视觉特征重要性
    visual_importance = np.var(visual_features, axis=0)
    top_visual_idx = np.argsort(visual_importance)[::-1][:15]

    axes[1, 0].bar(range(len(top_visual_idx)), visual_importance[top_visual_idx])
    axes[1, 0].set_title('视觉特征重要性 (Top 15)')
    axes[1, 0].set_xlabel('特征索引')
    axes[1, 0].set_ylabel('重要性分数')

    # 标注特征重要性
    annotation_importance = np.var(annotation_features, axis=0)
    top_annotation_idx = np.argsort(annotation_importance)[::-1][:15]

    axes[1, 1].bar(range(len(top_annotation_idx)), annotation_importance[top_annotation_idx])
    axes[1, 1].set_title('标注特征重要性 (Top 15)')
    axes[1, 1].set_xlabel('特征索引')
    axes[1, 1].set_ylabel('重要性分数')

    plt.tight_layout()
    plt.show()

    # 2. 模态间相关性分析
    plt.figure(figsize=(12, 8))

    # 计算各模态的平均特征
    audio_mean = np.mean(audio_features, axis=1)
    text_mean = np.mean(text_features, axis=1)
    visual_mean = np.mean(visual_features, axis=1)
    annotation_mean = np.mean(annotation_features, axis=1)

    modality_data = np.column_stack([audio_mean, text_mean, visual_mean, annotation_mean])
    correlation_matrix = np.corrcoef(modality_data.T)

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                xticklabels=['音频', '文本', '视觉', '标注'],
                yticklabels=['音频', '文本', '视觉', '标注'])
    plt.title('多模态特征相关性分析')
    plt.show()

    # 3. 类别分布可视化
    plt.figure(figsize=(15, 5))

    # 各模态在不同类别下的分布
    modalities = ['音频', '文本', '视觉', '标注']
    features_list = [audio_features, text_features, visual_features, annotation_features]

    for i, (modality, features) in enumerate(zip(modalities, features_list)):
        plt.subplot(1, 4, i + 1)

        true_features = features[labels == 0]
        false_features = features[labels == 1]

        plt.hist(np.mean(true_features, axis=1), alpha=0.7, label='真话', bins=20)
        plt.hist(np.mean(false_features, axis=1), alpha=0.7, label='假话', bins=20)

        plt.title(f'{modality}特征分布')
        plt.xlabel('特征值')
        plt.ylabel('频次')
        plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()