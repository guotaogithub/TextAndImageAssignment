from AudioFeatureExtractor import AudioFeatureExtractor
from VisualFeatureExtractor import VisualFeatureExtractor
import os
import glob
import numpy as np
import pandas as pd
from TextFeatureExtractor import TextFeatureExtractor


# ======================== Data Loading and Processing ========================
def _find_matching_files(directory, extensions):
    """Find matching files"""
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
        print(f"Error traversing directory {directory}: {e}")

    return sorted(matching_files)


class MultimodalDataLoader:
    """Multimodal data loader - Enhanced version"""

    def __init__(self, config):
        self.config = config
        self.audio_extractor = AudioFeatureExtractor()
        self.text_extractor = TextFeatureExtractor()
        self.visual_extractor = VisualFeatureExtractor()
        self.annotation_df = None

        # Load annotation data
        if os.path.exists(config.annotation_file):
            self.annotation_df = pd.read_csv(config.annotation_file)
            print(f"✅ Successfully loaded annotation file with {len(self.annotation_df)} records")
        else:
            print("⚠️ Annotation file not found")

    def load_annotation_features(self):
        """Load annotation features"""
        print("Loading annotation features...")

        if not hasattr(self.config, 'annotation_file') or not self.config.annotation_file:
            print("Annotation file path not configured")
            return None, None

        try:
            # Try to read annotation file
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
                print(f"Unsupported annotation file format: {self.config.annotation_file}")
                return None, None

            # Extract features and labels
            annotation_features = []
            annotation_labels = []

            # Assume annotation file contains the following columns:
            # - label: truth label (0/1 or False/True)
            # - confidence: confidence score
            # - emotion: emotion label
            # - other features...

            for idx, row in df.iterrows():
                feature_vector = []

                # Extract numeric features
                for col in df.columns:
                    if col.lower() not in ['label', 'filename', 'file', 'id']:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            feature_vector.append(row[col])
                        elif isinstance(row[col], str):
                            # Simple encoding for string features
                            feature_vector.append(hash(row[col]) % 1000)  # Simple hash encoding

                if feature_vector:
                    annotation_features.append(feature_vector)

                    # Extract labels
                    if 'label' in row:
                        label = row['label']
                        if isinstance(label, str):
                            label = 1 if label.lower() in ['true', '1', 'real'] else 0
                        annotation_labels.append(int(label))
                    else:
                        # If no label column, try to infer from filename
                        filename = row.get('filename', row.get('file', ''))
                        if 'true' in str(filename).lower() or 'real' in str(filename).lower():
                            annotation_labels.append(1)
                        else:
                            annotation_labels.append(0)

            if annotation_features:
                annotation_features = np.array(annotation_features)
                annotation_labels = np.array(annotation_labels)
                print(f"Annotation features loaded successfully: {annotation_features.shape}")
                return annotation_features, annotation_labels
            else:
                print("No valid features found in annotation file")
                return None, None

        except Exception as e:
            print(f"Error loading annotation file: {e}")
            return None, None

    def load_labels(self):
        """Load labels"""
        print("Integrating labels from all modalities...")

        all_labels = []

        # Collect labels from various modalities
        try:
            # Audio labels
            _, audio_labels = self.load_audio_features()
            if audio_labels is not None:
                all_labels.extend(audio_labels.tolist())

            # Text labels
            _, text_labels = self.load_text_features()
            if text_labels is not None:
                all_labels.extend(text_labels.tolist())

            # Visual labels
            _, visual_labels = self.load_visual_features()
            if visual_labels is not None:
                all_labels.extend(visual_labels.tolist())

            # Annotation labels
            _, annotation_labels = self.load_annotation_features()
            if annotation_labels is not None:
                all_labels.extend(annotation_labels.tolist())

            if all_labels:
                labels = np.array(all_labels)
                print(f"Labels loaded successfully: {len(labels)} samples")
                print(f"Label distribution - True: {np.sum(labels)}, Fake: {len(labels) - np.sum(labels)}")
                return labels
            else:
                print("No labels found")
                return None

        except Exception as e:
            print(f"Error loading labels: {e}")
            return None

    def _process_category(self, category, label):
        """Process data for a specific category - Enhanced version"""
        print(f"\n📁 Processing {'truth' if label == 0 else 'lie'} data...")

        category_data = []

        # Get video directory
        if category == "true":
            video_dir = self.config.video_true
        else:
            video_dir = self.config.video_false

        # Get video files
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov']:
            video_files.extend(glob.glob(os.path.join(video_dir, ext)))

        print(f"   Found {len(video_files)} video files")

        if len(video_files) == 0:
            print(f"   ⚠️ Video directory is empty: {video_dir}")
            return category_data

        for i, video_file in enumerate(video_files):
            try:
                filename = os.path.basename(video_file)
                file_id = filename.replace('.mp4', '').replace('.avi', '').replace('.mov', '')

                print(f"   Processing file {i + 1}/{len(video_files)}: {filename}")

                # 1. Smart search for matching text and audio files
                text_content, audio_file = _find_matching_files(video_file, category)

                # 2. Extract text features
                text_features = self.text_extractor.extract_text_features(text_content)

                # 3. Extract audio features
                if audio_file:
                    audio_features = self.audio_extractor.extract_comprehensive_features(audio_file)
                else:
                    audio_features = np.zeros(64)
                    print(f"      ⚠️ Using zero audio features")

                # 4. Extract visual features
                print(f"      🎥 Extracting visual features...")
                visual_features = self.visual_extractor.extract_video_features(video_file)

                # 5. Get annotation features
                annotation_features = self._get_annotation_features(filename)

                # Combine data
                sample_data = {
                    'file_id': file_id,
                    'audio_features': audio_features,
                    'text_features': text_features,
                    'visual_features': visual_features,
                    'annotation_features': annotation_features,
                    'label': label,
                    'category': category,
                    'text_content': text_content  # Save text content for further analysis
                }

                category_data.append(sample_data)
                print(f"      ✅ Sample processing completed")

            except Exception as e:
                print(f"   ❌ Error processing file {filename}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"   ✅ Successfully processed {len(category_data)} samples")
        return category_data

    def load_all_data(self):
        """Load all data"""
        print("🔄 Starting to load all data...")

        try:
            # Load audio features
            print("🔄 Loading audio features...")
            audio_data = self.load_audio_features()
            if audio_data[0] is not None:
                audio_features, audio_labels = audio_data
                print(f"✅ Audio features loaded successfully, shape: {audio_features.shape}")
            else:
                audio_features, audio_labels = None, None
                print("⚠️ Failed to load audio features")

            # Load text features
            print("🔄 Loading text features...")
            text_data = self.load_text_features()
            if text_data[0] is not None:
                text_features, text_labels = text_data
                print(f"✅ Text features loaded successfully, shape: {text_features.shape}")
            else:
                text_features, text_labels = None, None
                print("⚠️ Failed to load text features")

            # Load visual features
            print("🔄 Loading visual features...")
            visual_data = self.load_visual_features()
            if visual_data[0] is not None:
                visual_features, visual_labels = visual_data
                print(f"✅ Visual features loaded successfully, shape: {visual_features.shape}")
            else:
                visual_features, visual_labels = None, None
                print("⚠️ Failed to load visual features")

            # Load annotation features
            print("🔄 Loading annotation features...")
            annotation_data = self.load_annotation_features()
            if annotation_data[0] is not None:
                annotation_features, annotation_labels = annotation_data
                print(f"✅ Annotation features loaded successfully, shape: {annotation_features.shape}")
            else:
                annotation_features, annotation_labels = None, None
                print("⚠️ Failed to load annotation features")

            # Collect all valid features and labels
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
                print("❌ No features were successfully loaded")
                return None

            print(f"✅ Successfully loaded {len(all_features)} modalities of features: {feature_names}")

            # Return data in dictionary format
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
            print(f"❌ Error occurred during data loading: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_audio_features(self):
        """Load audio features"""
        print("Loading audio features...")
        audio_features = []
        audio_labels = []

        # Process real audio files
        if hasattr(self.config, 'audio_true') and self.config.audio_true:
            true_files = _find_matching_files(
                self.config.audio_true,
                ['.wav', '.mp3', '.m4a', '.flac']
            )
            print(f"Found {len(true_files)} real audio files")

            for file_path in true_files:
                try:
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        features = self.audio_extractor.extract_comprehensive_features(file_path)
                        if features is not None and len(features) > 0:
                            audio_features.append(features)
                            audio_labels.append(1)  # Real label
                    else:
                        print(f"File does not exist or is empty: {file_path}")
                except Exception as e:
                    print(f"Error processing audio file {file_path}: {e}")

        # Process fake audio files
        if hasattr(self.config, 'audio_false') and self.config.audio_false:
            false_files = _find_matching_files(
                self.config.audio_false,
                ['.wav', '.mp3', '.m4a', '.flac']
            )
            print(f"Found {len(false_files)} fake audio files")

            for file_path in false_files:
                try:
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        features = self.audio_extractor.extract_comprehensive_features(file_path)
                        if features is not None and len(features) > 0:
                            audio_features.append(features)
                            audio_labels.append(0)  # Fake label
                    else:
                        print(f"File does not exist or is empty: {file_path}")
                except Exception as e:
                    print(f"Error processing audio file {file_path}: {e}")

        # Process edited fake audio files
        if hasattr(self.config, 'audio_false_edited') and self.config.audio_false_edited:
            false_edited_files = _find_matching_files(
                self.config.audio_false_edited,
                ['.wav', '.mp3', '.m4a', '.flac']
            )
            print(f"Found {len(false_edited_files)} edited fake audio files")

            for file_path in false_edited_files:
                try:
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        features = self.audio_extractor.extract_comprehensive_features(file_path)
                        if features is not None and len(features) > 0:
                            audio_features.append(features)
                            audio_labels.append(0)  # Fake label
                    else:
                        print(f"File does not exist or is empty: {file_path}")
                except Exception as e:
                    print(f"Error processing audio file {file_path}: {e}")

        if audio_features:
            audio_features = np.array(audio_features)
            audio_labels = np.array(audio_labels)
            print(f"Audio features loaded successfully: {audio_features.shape}")
            return audio_features, audio_labels
        else:
            print("No valid audio features found")
            return None, None

    def load_text_features(self):
        """Load text features"""
        print("Loading text features...")
        text_features = []
        text_labels = []

        # Process real text files
        if hasattr(self.config, 'text_true') and self.config.text_true:
            true_files = _find_matching_files(
                self.config.text_true,
                ['.txt', '.csv', '.json']
            )
            print(f"Found {len(true_files)} real text files")

            for file_path in true_files:
                try:
                    if not os.path.exists(file_path):
                        continue

                    # Read text content
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        # Assume text is in one column, try common column names
                        text_columns = ['text', 'content', 'transcript', 'sentence']
                        text_column = None
                        for col in text_columns:
                            if col in df.columns:
                                text_column = col
                                break
                        if text_column:
                            texts = df[text_column].dropna().tolist()
                        else:
                            texts = df.iloc[:, 0].dropna().tolist()  # Use first column
                    elif file_path.endswith('.json'):
                        import json
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            texts = [str(item) for item in data]
                        elif isinstance(data, dict):
                            # Try common key names
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
                    else:  # .txt file
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
                                text_labels.append(1)  # Real label
                except Exception as e:
                    print(f"Error processing text file {file_path}: {e}")

        # Process fake text files
        if hasattr(self.config, 'text_false') and self.config.text_false:
            false_files = _find_matching_files(
                self.config.text_false,
                ['.txt', '.csv', '.json']
            )
            print(f"Found {len(false_files)} fake text files")

            for file_path in false_files:
                try:
                    if not os.path.exists(file_path):
                        continue

                    # Read text content (same logic as above)
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
                                text_labels.append(0)  # Fake label
                except Exception as e:
                    print(f"Error processing text file {file_path}: {e}")

        if text_features:
            text_features = np.array(text_features)
            text_labels = np.array(text_labels)
            print(f"Text features loaded successfully: {text_features.shape}")
            return text_features, text_labels
        else:
            print("No valid text features found")
            return None, None

    def load_visual_features(self):
        """Load visual features"""
        print("Loading visual features...")
        visual_features = []
        visual_labels = []

        # Process real video files
        if hasattr(self.config, 'video_true') and self.config.video_true:
            true_files = _find_matching_files(
                self.config.video_true,
                ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
            )
            print(f"Found {len(true_files)} real video files")

            for file_path in true_files:
                try:
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        features = self.visual_extractor.extract_video_features(file_path)
                        if features is not None and len(features) > 0:
                            # If multiple frames are returned, take average
                            if isinstance(features, list):
                                if len(features) > 0:
                                    mean_features = np.mean(features, axis=0)
                                    visual_features.append(mean_features)
                                    visual_labels.append(1)  # Real label
                            else:
                                visual_features.append(features)
                                visual_labels.append(1)  # Real label
                    else:
                        print(f"File does not exist or is empty: {file_path}")
                except Exception as e:
                    print(f"Error processing video file {file_path}: {e}")

        # Process fake video files
        if hasattr(self.config, 'video_false') and self.config.video_false:
            false_files = _find_matching_files(
                self.config.video_false,
                ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
            )
            print(f"Found {len(false_files)} fake video files")

            for file_path in false_files:
                try:
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                        features = self.visual_extractor.extract_video_features(file_path)
                        if features is not None and len(features) > 0:
                            # If multiple frames are returned, take average
                            if isinstance(features, list):
                                if len(features) > 0:
                                    mean_features = np.mean(features, axis=0)
                                    visual_features.append(mean_features)
                                    visual_labels.append(0)  # Fake label
                            else:
                                visual_features.append(features)
                                visual_labels.append(0)  # Fake label
                    else:
                        print(f"File does not exist or is empty: {file_path}")
                except Exception as e:
                    print(f"Error processing video file {file_path}: {e}")

        if visual_features:
            visual_features = np.array(visual_features)
            visual_labels = np.array(visual_labels)
            print(f"Visual features loaded successfully: {visual_features.shape}")
            return visual_features, visual_labels
        else:
            print("No valid visual features found")
            return None, None
