import numpy as np
import cv2
import mediapipe as mp


# ======================== Visual Feature Extractor ========================
class VisualFeatureExtractor:
    """Visual feature extractor"""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

    def extract_video_features(self, video_path):
        """Extract visual features from a video"""
        try:
            cap = cv2.VideoCapture(video_path)

            face_landmarks_data = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % 5 != 0:  # Sample every 5 frames
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0]
                    # Extract key facial landmark points
                    landmark_coords = []
                    for landmark in landmarks.landmark:
                        landmark_coords.extend([landmark.x, landmark.y])
                    face_landmarks_data.append(landmark_coords)

            cap.release()

            if face_landmarks_data:
                # Calculate statistical features of facial landmarks
                landmarks_array = np.array(face_landmarks_data)
                features = [
                    np.mean(landmarks_array, axis=0),
                    np.std(landmarks_array, axis=0),
                    np.max(landmarks_array, axis=0) - np.min(landmarks_array, axis=0)
                ]
                return np.concatenate(features)[:300]  # Limit to 300 dimensions
            else:
                return np.zeros(300)

        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return np.zeros(300)
