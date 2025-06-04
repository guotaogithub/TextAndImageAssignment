import os
import cv2
import numpy as np
import mediapipe as mp


class VisualFeatureExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def extract_video_features(self, video_path):
        """Extract visual features from a video without emotion analysis"""
        try:
            if not os.path.exists(video_path):
                return np.zeros(300)

            cap = cv2.VideoCapture(video_path)
            face_landmarks_data = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % 5 != 0:  # 每隔几帧处理一次
                    continue

                # 提取面部关键点特征
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    landmark_coords = []
                    for landmark in landmarks.landmark:
                        landmark_coords.extend([landmark.x, landmark.y])
                    face_landmarks_data.append(landmark_coords)

            cap.release()

            # 处理地标特征
            if face_landmarks_data:
                landmarks_array = np.array(face_landmarks_data)

                # 统计特征：均值、标准差、极差
                face_features = [
                    np.mean(landmarks_array, axis=0),
                    np.std(landmarks_array, axis=0),
                    np.max(landmarks_array, axis=0) - np.min(landmarks_array, axis=0)
                ]

                combined_features = np.concatenate([
                    np.concatenate(face_features)
                ])[:300]

                return combined_features
            else:
                return np.zeros(300)

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return np.zeros(300)
