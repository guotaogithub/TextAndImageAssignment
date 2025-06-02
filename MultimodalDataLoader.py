from AudioFeatureExtractor import AudioFeatureExtractor
from VisualFeatureExtractor import VisualFeatureExtractor
import os
import glob
import numpy as np
import pandas as pd
from TextFeatureExtractor import TextFeatureExtractor


# ======================== 数据加载和处理 ========================
def _find_matching_files(directory, extensions):
    """查找匹配的文件"""
    if not directory or not os.path.exists(directory):
        return []

    matching_files = []
    try:
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext.lower()) for ext in extensions):
                full_path = os.path.join(directory, file)
                if os.path.isfile(full_path):
                    matching_files.append(full_path)
    except Exception as e:
        print(f"遍历目录 {directory} 时出错: {e}")

    return sorted(matching_files)


class MultimodalDataLoader:
    """多模态数据加载器 - 增强版"""

    def __init__(self, config):
        self.config = config
        self.audio_extractor = AudioFeatureExtractor()
        self.text_extractor = TextFeatureExtractor()
        self.visual_extractor = VisualFeatureExtractor()
        self.annotation_df = None

        # 加载标注数据
        if os.path.exists(config.annotation_file):
            self.annotation_df = pd.read_csv(config.annotation_file)
            print(f"✅ 成功加载标注文件，共 {len(self.annotation_df)} 条记录")
        else:
            print("⚠️ 未找到标注文件")

    def load_annotation_features(self):
        """加载标注特征"""
        """加载标注特征"""
        print("正在加载标注特征...")

        if not hasattr(self.config, 'annotation_file') or not self.config.annotation_file:
            print("未配置标注文件路径")
            return None, None

        try:
            # 尝试读取标注文件
            if self.config.annotation_file.endswith('.csv'):
                df = pd.read_csv(self.config.annotation_file)
            elif self.config.annotation_file.endswith('.json'):
                import json
                with open(self.config.annotation_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame([data])
            else:
                print(f"不支持的标注文件格式: {self.config.annotation_file}")
                return None, None

            # 提取特征和标签
            annotation_features = []
            annotation_labels = []

            # 假设标注文件包含以下列：
            # - label: 真实性标签 (0/1 或 False/True)
            # - confidence: 置信度
            # - emotion: 情感标签
            # - other features...

            for idx, row in df.iterrows():
                feature_vector = []

                # 提取数值特征
                for col in df.columns:
                    if col.lower() not in ['label', 'filename', 'file', 'id']:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            feature_vector.append(row[col])
                        elif isinstance(row[col], str):
                            # 对字符串特征进行简单编码
                            feature_vector.append(hash(row[col]) % 1000)  # 简单哈希编码

                if feature_vector:
                    annotation_features.append(feature_vector)

                    # 提取标签
                    if 'label' in row:
                        label = row['label']
                        if isinstance(label, str):
                            label = 1 if label.lower() in ['true', '1', 'real'] else 0
                        annotation_labels.append(int(label))
                    else:
                        # 如果没有标签列，尝试从文件名推断
                        filename = row.get('filename', row.get('file', ''))
                        if 'true' in str(filename).lower() or 'real' in str(filename).lower():
                            annotation_labels.append(1)
                        else:
                            annotation_labels.append(0)

            if annotation_features:
                annotation_features = np.array(annotation_features)
                annotation_labels = np.array(annotation_labels)
                print(f"标注特征加载完成: {annotation_features.shape}")
                return annotation_features, annotation_labels
            else:
                print("标注文件中未找到有效特征")
                return None, None

        except Exception as e:
            print(f"加载标注文件时出错: {e}")
            return None, None

    def load_labels(self):
        """加载标签"""
        print("正在整合所有模态的标签...")

        all_labels = []

        # 从各个模态收集标签
        try:
            # 音频标签
            _, audio_labels = self.load_audio_features()
            if audio_labels is not None:
                all_labels.extend(audio_labels.tolist())

            # 文本标签
            _, text_labels = self.load_text_features()
            if text_labels is not None:
                all_labels.extend(text_labels.tolist())

            # 视觉标签
            _, visual_labels = self.load_visual_features()
            if visual_labels is not None:
                all_labels.extend(visual_labels.tolist())

            # 标注标签
            _, annotation_labels = self.load_annotation_features()
            if annotation_labels is not None:
                all_labels.extend(annotation_labels.tolist())

            if all_labels:
                labels = np.array(all_labels)
                print(f"标签加载完成: {len(labels)} 个样本")
                print(f"标签分布 - 真实: {np.sum(labels)}, 虚假: {len(labels) - np.sum(labels)}")
                return labels
            else:
                print("未找到任何标签")
                return None

        except Exception as e:
            print(f"加载标签时出错: {e}")
            return None

    def _process_category(self, category, label):
        """处理特定类别的数据 - 增强版"""
        print(f"\n📁 处理 {'真话' if label == 0 else '假话'} 数据...")

        category_data = []

        # 获取视频目录
        if category == "true":
            video_dir = self.config.video_true
        else:
            video_dir = self.config.video_false

        # 获取视频文件
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov']:
            video_files.extend(glob.glob(os.path.join(video_dir, ext)))

        print(f"   找到 {len(video_files)} 个视频文件")

        if len(video_files) == 0:
            print(f"   ⚠️ 视频目录为空: {video_dir}")
            return category_data

        for i, video_file in enumerate(video_files):
            try:
                filename = os.path.basename(video_file)
                file_id = filename.replace('.mp4', '').replace('.avi', '').replace('.mov', '')

                print(f"   处理文件 {i + 1}/{len(video_files)}: {filename}")

                # 1. 智能查找匹配的文本和音频文件
                text_content, audio_file = _find_matching_files(video_file, category)

                # 2. 提取文本特征
                text_features = self.text_extractor.extract_text_features(text_content)

                # 3. 提取音频特征
                if audio_file:
                    audio_features = self.audio_extractor.extract_comprehensive_features(audio_file)
                else:
                    audio_features = np.zeros(64)
                    print(f"      ⚠️ 使用零音频特征")

                # 4. 提取视觉特征
                print(f"      🎥 提取视觉特征...")
                visual_features = self.visual_extractor.extract_video_features(video_file)

                # 5. 获取标注特征
                annotation_features = self._get_annotation_features(filename)

                # 组合数据
                sample_data = {
                    'file_id': file_id,
                    'audio_features': audio_features,
                    'text_features': text_features,
                    'visual_features': visual_features,
                    'annotation_features': annotation_features,
                    'label': label,
                    'category': category,
                    'text_content': text_content  # 保存文本内容用于后续分析
                }

                category_data.append(sample_data)
                print(f"      ✅ 样本处理完成")

            except Exception as e:
                print(f"   ❌ 处理文件 {filename} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"   ✅ 成功处理 {len(category_data)} 个样本")
        return category_data

    def load_all_data(self):
        """加载所有数据"""
        print("🔄 开始加载所有数据...")

        try:
            # 加载音频特征
            print("🔄 开始加载音频特征...")
            audio_data = self.load_audio_features()
            if audio_data[0] is not None:
                audio_features, audio_labels = audio_data
                print(f"✅ 音频特征加载完成，形状: {audio_features.shape}")
            else:
                audio_features, audio_labels = None, None
                print("⚠️ 音频特征加载失败")

            # 加载文本特征
            print("🔄 开始加载文本特征...")
            text_data = self.load_text_features()
            if text_data[0] is not None:
                text_features, text_labels = text_data
                print(f"✅ 文本特征加载完成，形状: {text_features.shape}")
            else:
                text_features, text_labels = None, None
                print("⚠️ 文本特征加载失败")

            # 加载视觉特征
            print("🔄 开始加载视觉特征...")
            visual_data = self.load_visual_features()
            if visual_data[0] is not None:
                visual_features, visual_labels = visual_data
                print(f"✅ 视觉特征加载完成，形状: {visual_features.shape}")
            else:
                visual_features, visual_labels = None, None
                print("⚠️ 视觉特征加载失败")

            # 加载标注特征
            print("🔄 开始加载标注特征...")
            annotation_data = self.load_annotation_features()
            if annotation_data[0] is not None:
                annotation_features, annotation_labels = annotation_data
                print(f"✅ 标注特征加载完成，形状: {annotation_features.shape}")
            else:
                annotation_features, annotation_labels = None, None
                print("⚠️ 标注特征加载失败")

            # 收集所有有效的特征和标签
            all_features = []
            all_labels = []
            feature_names = []

            if audio_features is not None:
                all_features.append(audio_features)
                all_labels.append(audio_labels)
                feature_names.append('audio')

            if text_features is not None:
                all_features.append(text_features)
                all_labels.append(text_labels)
                feature_names.append('text')

            if visual_features is not None:
                all_features.append(visual_features)
                all_labels.append(visual_labels)
                feature_names.append('visual')

            if annotation_features is not None:
                all_features.append(annotation_features)
                all_labels.append(annotation_labels)
                feature_names.append('annotation')

            if not all_features:
                print("❌ 没有成功加载任何特征")
                return None

            print(f"✅ 成功加载 {len(all_features)} 种模态的特征: {feature_names}")

            # 返回字典格式的数据
            return {
                'audio_features': audio_features,
                'text_features': text_features,
                'visual_features': visual_features,
                'annotation_features': annotation_features,
                'audio_labels': audio_labels,
                'text_labels': text_labels,
                'visual_labels': visual_labels,
                'annotation_labels': annotation_labels,
                'feature_names': feature_names
            }

        except Exception as e:
            print(f"❌ 数据加载过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_audio_features(self):
        """加载音频特征"""
        print("正在加载音频特征...")
        audio_features = []
        audio_labels = []

        # 处理真实语音文件
        if hasattr(self.config, 'audio_true') and self.config.audio_true:
            true_files = _find_matching_files(
                self.config.audio_true,
                ['.wav', '.mp3', '.m4a', '.flac']
            )
            print(f"找到 {len(true_files)} 个真实音频文件")

            for file_path in true_files:
                try:
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        features = self.audio_extractor.extract_comprehensive_features(file_path)
                        if features is not None and len(features) > 0:
                            audio_features.append(features)
                            audio_labels.append(1)  # 真实标签
                    else:
                        print(f"文件不存在或为空: {file_path}")
                except Exception as e:
                    print(f"处理音频文件 {file_path} 时出错: {e}")

        # 处理虚假语音文件
        if hasattr(self.config, 'audio_false') and self.config.audio_false:
            false_files = _find_matching_files(
                self.config.audio_false,
                ['.wav', '.mp3', '.m4a', '.flac']
            )
            print(f"找到 {len(false_files)} 个虚假音频文件")

            for file_path in false_files:
                try:
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        features = self.audio_extractor.extract_comprehensive_features(file_path)
                        if features is not None and len(features) > 0:
                            audio_features.append(features)
                            audio_labels.append(0)  # 虚假标签
                    else:
                        print(f"文件不存在或为空: {file_path}")
                except Exception as e:
                    print(f"处理音频文件 {file_path} 时出错: {e}")

        # 处理编辑过的虚假语音文件
        if hasattr(self.config, 'audio_false_edited') and self.config.audio_false_edited:
            false_edited_files = _find_matching_files(
                self.config.audio_false_edited,
                ['.wav', '.mp3', '.m4a', '.flac']
            )
            print(f"找到 {len(false_edited_files)} 个编辑过的虚假音频文件")

            for file_path in false_edited_files:
                try:
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        features = self.audio_extractor.extract_comprehensive_features(file_path)
                        if features is not None and len(features) > 0:
                            audio_features.append(features)
                            audio_labels.append(0)  # 虚假标签
                    else:
                        print(f"文件不存在或为空: {file_path}")
                except Exception as e:
                    print(f"处理音频文件 {file_path} 时出错: {e}")

        if audio_features:
            audio_features = np.array(audio_features)
            audio_labels = np.array(audio_labels)
            print(f"音频特征加载完成: {audio_features.shape}")
            return audio_features, audio_labels
        else:
            print("未找到有效的音频特征")
            return None, None

    def load_text_features(self):
        """加载文本特征"""
        print("正在加载文本特征...")
        text_features = []
        text_labels = []

        # 处理真实文本文件
        if hasattr(self.config, 'text_true') and self.config.text_true:
            true_files = _find_matching_files(
                self.config.text_true,
                ['.txt', '.csv', '.json']
            )
            print(f"找到 {len(true_files)} 个真实文本文件")

            for file_path in true_files:
                try:
                    if not os.path.exists(file_path):
                        continue

                    # 读取文本内容
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        # 假设文本在某一列中，尝试常见的列名
                        text_columns = ['text', 'content', 'transcript', 'sentence']
                        text_column = None
                        for col in text_columns:
                            if col in df.columns:
                                text_column = col
                                break
                        if text_column:
                            texts = df[text_column].dropna().tolist()
                        else:
                            texts = df.iloc[:, 0].dropna().tolist()  # 使用第一列
                    elif file_path.endswith('.json'):
                        import json
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            texts = [str(item) for item in data]
                        elif isinstance(data, dict):
                            # 尝试常见的键名
                            text_keys = ['text', 'content', 'transcript', 'sentence']
                            texts = []
                            for key in text_keys:
                                if key in data:
                                    texts = data[key] if isinstance(data[key], list) else [data[key]]
                                    break
                            if not texts:
                                texts = [str(data)]
                        else:
                            texts = [str(data)]
                    else:  # .txt文件
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content:
                                texts = [content]
                            else:
                                texts = []

                    for text in texts:
                        if text and len(str(text).strip()) > 0:
                            features = self.text_extractor.extract_text_features(str(text))
                            if features is not None and len(features) > 0:
                                text_features.append(features)
                                text_labels.append(1)  # 真实标签
                except Exception as e:
                    print(f"处理文本文件 {file_path} 时出错: {e}")

        # 处理虚假文本文件
        if hasattr(self.config, 'text_false') and self.config.text_false:
            false_files = _find_matching_files(
                self.config.text_false,
                ['.txt', '.csv', '.json']
            )
            print(f"找到 {len(false_files)} 个虚假文本文件")

            for file_path in false_files:
                try:
                    if not os.path.exists(file_path):
                        continue

                    # 读取文本内容（与上面相同的逻辑）
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        text_columns = ['text', 'content', 'transcript', 'sentence']
                        text_column = None
                        for col in text_columns:
                            if col in df.columns:
                                text_column = col
                                break
                        if text_column:
                            texts = df[text_column].dropna().tolist()
                        else:
                            texts = df.iloc[:, 0].dropna().tolist()
                    elif file_path.endswith('.json'):
                        import json
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            texts = [str(item) for item in data]
                        elif isinstance(data, dict):
                            text_keys = ['text', 'content', 'transcript', 'sentence']
                            texts = []
                            for key in text_keys:
                                if key in data:
                                    texts = data[key] if isinstance(data[key], list) else [data[key]]
                                    break
                            if not texts:
                                texts = [str(data)]
                        else:
                            texts = [str(data)]
                    else:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            if content:
                                texts = [content]
                            else:
                                texts = []

                    for text in texts:
                        if text and len(str(text).strip()) > 0:
                            features = self.text_extractor.extract_text_features(str(text))
                            if features is not None and len(features) > 0:
                                text_features.append(features)
                                text_labels.append(0)  # 虚假标签
                except Exception as e:
                    print(f"处理文本文件 {file_path} 时出错: {e}")

        if text_features:
            text_features = np.array(text_features)
            text_labels = np.array(text_labels)
            print(f"文本特征加载完成: {text_features.shape}")
            return text_features, text_labels
        else:
            print("未找到有效的文本特征")
            return None, None

    def load_visual_features(self):
        """加载视觉特征"""
        print("正在加载视觉特征...")
        visual_features = []
        visual_labels = []

        # 处理真实视频文件
        if hasattr(self.config, 'video_true') and self.config.video_true:
            true_files = _find_matching_files(
                self.config.video_true,
                ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
            )
            print(f"找到 {len(true_files)} 个真实视频文件")

            for file_path in true_files:
                try:
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        features = self.visual_extractor.extract_video_features(file_path)
                        if features is not None and len(features) > 0:
                            # 如果返回多帧特征，取平均值
                            if isinstance(features, list):
                                if len(features) > 0:
                                    mean_features = np.mean(features, axis=0)
                                    visual_features.append(mean_features)
                                    visual_labels.append(1)  # 真实标签
                            else:
                                visual_features.append(features)
                                visual_labels.append(1)  # 真实标签
                    else:
                        print(f"文件不存在或为空: {file_path}")
                except Exception as e:
                    print(f"处理视频文件 {file_path} 时出错: {e}")

        # 处理虚假视频文件
        if hasattr(self.config, 'video_false') and self.config.video_false:
            false_files = _find_matching_files(
                self.config.video_false,
                ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
            )
            print(f"找到 {len(false_files)} 个虚假视频文件")

            for file_path in false_files:
                try:
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        features = self.visual_extractor.extract_video_features(file_path)
                        if features is not None and len(features) > 0:
                            # 如果返回多帧特征，取平均值
                            if isinstance(features, list):
                                if len(features) > 0:
                                    mean_features = np.mean(features, axis=0)
                                    visual_features.append(mean_features)
                                    visual_labels.append(0)  # 虚假标签
                            else:
                                visual_features.append(features)
                                visual_labels.append(0)  # 虚假标签
                    else:
                        print(f"文件不存在或为空: {file_path}")
                except Exception as e:
                    print(f"处理视频文件 {file_path} 时出错: {e}")

        if visual_features:
            visual_features = np.array(visual_features)
            visual_labels = np.array(visual_labels)
            print(f"视觉特征加载完成: {visual_features.shape}")
            return visual_features, visual_labels
        else:
            print("未找到有效的视觉特征")
            return None, None


