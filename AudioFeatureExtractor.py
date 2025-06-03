import librosa
import numpy as np


# ======================== Audio Feature Extractor ========================
class AudioFeatureExtractor:
    """Audio feature extractor"""

    def __init__(self, sr=22050, n_mfcc=13):
        self.sr = sr
        self.n_mfcc = n_mfcc

    def extract_comprehensive_features(self, audio_path):
        """Extract comprehensive audio features"""
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sr)

            # 1. MFCC features (13 coefficients × 4 statistics = 52 features)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_features = []
            for i in range(mfccs.shape[0]):
                mfcc_features.extend([
                    np.mean(mfccs[i]), np.std(mfccs[i]),
                    np.max(mfccs[i]), np.min(mfccs[i])
                ])

            # 2. Pitch features (6 features)
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                                         fmax=librosa.note_to_hz('C7'))
            f0_clean = f0[~np.isnan(f0)]
            if len(f0_clean) > 0:
                pitch_features = [
                    np.mean(f0_clean), np.std(f0_clean),
                    np.max(f0_clean), np.min(f0_clean),
                    np.percentile(f0_clean, 75) - np.percentile(f0_clean, 25),  # IQR
                    np.sum(voiced_flag) / len(voiced_flag)  # Voiced ratio
                ]
            else:
                pitch_features = [0, 0, 0, 0, 0, 0]

            # 3. Energy features (6 features)
            rms = librosa.feature.rms(y=y)[0]
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            zcr = librosa.feature.zero_crossing_rate(y)[0]

            energy_features = [
                np.mean(rms), np.std(rms),
                np.mean(spectral_centroids), np.std(spectral_centroids),
                np.mean(zcr), np.std(zcr)
            ]

            # Combine all features
            all_features = mfcc_features + pitch_features + energy_features
            return np.array(all_features)

        except Exception as e:
            print(f"Error extracting audio features from {audio_path}: {e}")
            return np.zeros(64)  # Return zero vector
