import numpy as np
import cv2
import mediapipe as mp


# ======================== 视觉特征提取器 ========================
class VisualFeatureExtractor:
    """视觉特征提取器"""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def extract_video_features(self, video_path):
        """从视频中提取视觉特征"""
        try:
            cap = cv2.VideoCapture(video_path)

            face_landmarks_data = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % 5 != 0:  # 每5帧采样一次
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    # 提取关键面部标志点
                    landmark_coords = []
                    for landmark in landmarks.landmark:
                        landmark_coords.extend([landmark.x, landmark.y])
                    face_landmarks_data.append(landmark_coords)

            cap.release()

            if face_landmarks_data:
                # 计算面部标志点的统计特征
                landmarks_array = np.array(face_landmarks_data)
                features = [
                    np.mean(landmarks_array, axis=0),
                    np.std(landmarks_array, axis=0),
                    np.max(landmarks_array, axis=0) - np.min(landmarks_array, axis=0)
                ]
                return np.concatenate(features)[:300]  # 限制为300维
            else:
                return np.zeros(300)

        except Exception as e:
            print(f"处理视频 {video_path} 时出错: {e}")
            return np.zeros(300)
