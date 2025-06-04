from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
from MultimodalFusionModel import MultimodalFusionModel
from TextFeatureExtractor import TextFeatureExtractor
from AudioFeatureExtractor import AudioFeatureExtractor
from VisualFeatureExtractor import VisualFeatureExtractor
import tempfile
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# 加载模型
model_path = 'attention_fusion_model.pth'
model = MultimodalFusionModel(
    audio_dim=64,
    text_dim=768,
    visual_dim=300,
    annotation_dim=40,
    fusion_type='attention'
)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# 初始化特征提取器
text_extractor = TextFeatureExtractor()
audio_extractor = AudioFeatureExtractor()
visual_extractor = VisualFeatureExtractor()

def ensure_array(features, dim):
    """确保特征是正确维度"""
    if features is None:
        return np.zeros(dim)
    elif isinstance(features, list):
        features = np.array(features)

    if features.shape != (dim,):
        try:
            if features.size < dim:
                pad_width = ((0, dim - features.size),)
                features = np.pad(features, pad_width, mode='constant')
            else:
                features = features[:dim]
        except Exception as e:
            print(f"Error resizing features: {e}, using zero vector")
            return np.zeros(dim)

    return features


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        files = request.files.getlist('file')
        if not files or len(files) == 0:
            return jsonify({'error': 'No file uploaded'}), 400

        valid_extensions = {
            'text': ['.txt'],
            'audio': ['.wav', '.mp3', '.ogg', '.flac'],
            'video': ['.mp4', '.avi', '.mov', '.mkv']
        }

        exts = []
        for file in files:
            _, ext = os.path.splitext(file.filename.lower())
            exts.append(ext)

        unique_exts = list(set(exts))

        # 校验扩展名
        for ext in exts:
            if ext not in [e for exts_list in valid_extensions.values() for e in exts_list]:
                return jsonify({'error': f'Unsupported file type: {ext}'}), 400

        # 不允许上传相同类型的文件
        if len(exts) != len(unique_exts):
            return jsonify({'error': 'You cannot upload two files of the same type.'}), 400

        # 归类为 modality
        modalities = set()
        for ext in exts:
            for key, exts_list in valid_extensions.items():
                if ext in exts_list:
                    modalities.add(key)

        allowed_combinations = [
            {'text'}, {'audio'}, {'video'},
            {'text', 'audio'}, {'text', 'video'}, {'audio', 'video'},
            {'text', 'audio', 'video'}
        ]

        if modalities not in allowed_combinations:
            return jsonify({
                'error': 'Invalid combination. Please follow the rules:<br>'
                         '1. One single file<br>'
                         '2. Two different types<br>'
                         '3. All three types'
            }), 400

        # 初始化所有特征为 None
        text_features = None
        audio_features = None
        visual_features = None

        # 处理每个文件
        for file in files:
            filename = file.filename
            _, ext = os.path.splitext(filename.lower())

            with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
                file.save(tmpfile.name)
                temp_path = tmpfile.name

                if ext in valid_extensions['text']:
                    content = file.read().decode('utf-8')
                    text_features = text_extractor.extract_text_features(content)
                elif ext in valid_extensions['audio']:
                    audio_features = audio_extractor.extract_comprehensive_features(temp_path)
                elif ext in valid_extensions['video']:
                    visual_features = visual_extractor.extract_video_features(temp_path)

                os.unlink(temp_path)

        # 获取注解特征维度
        annotation_dim = model.annotation_proj[0].in_features
        annotation_features = np.zeros(annotation_dim)

        # 确保所有特征维度正确
        text_features = ensure_array(text_features, 768)
        audio_features = ensure_array(audio_features, 64)
        visual_features = ensure_array(visual_features, 300)
        annotation_features = ensure_array(annotation_features, annotation_dim)

        # 转换为张量
        audio_tensor = torch.FloatTensor(audio_features).unsqueeze(0)
        text_tensor = torch.FloatTensor(text_features).unsqueeze(0)
        visual_tensor = torch.FloatTensor(visual_features).unsqueeze(0)
        annotation_tensor = torch.FloatTensor(annotation_features).unsqueeze(0)

        # 推理
        with torch.no_grad():
            output = model(audio_tensor, text_tensor, visual_tensor, annotation_tensor)
            probabilities = torch.softmax(output, dim=1).squeeze().numpy()
            prediction = int(np.argmax(probabilities))

        # 构建返回数据
        result = {
            'prediction': 'Truth' if prediction == 0 else 'Lie',
            'confidence': float(probabilities[prediction]),
            'probabilities': probabilities.tolist()
        }

        # 如果需要返回 F1 分数和混淆矩阵（可选）
        # 注意：你需要有 val_loader 和 model_trainer 的定义才能启用这部分
        # 下面是一个占位符实现
        # cm_display = ConfusionMatrixDisplay(confusion_matrix=np.array([[5, 1], [1, 5]]))
        # fig, ax = plt.subplots(figsize=(4, 4))
        # cm_display.plot(ax=ax, cmap='Blues', values_format='')
        # buf = BytesIO()
        # plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        # plt.close()
        # data = base64.b64encode(buf.getvalue()).decode('utf-8')
        # result['confusion_matrix'] = data

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
