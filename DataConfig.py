import os

# ======================== Data path configuration ========================
class DataConfig:
    """Data path configuration class"""

    def __init__(self):
        self.base_path = os.path.join(os.path.dirname(__file__), "dataset")

        # Video paths
        self.video_false = os.path.join(self.base_path, "Clips/false")
        self.video_true = os.path.join(self.base_path, "Clips/true")

        # Text paths
        self.text_false = os.path.join(self.base_path, "Transcription/false")
        self.text_true = os.path.join(self.base_path, "Transcription/true")

        # Audio paths
        self.audio_false = os.path.join(self.base_path, "audio/false")
        self.audio_true = os.path.join(self.base_path, "audio/true")

        # Annotation file
        self.annotation_file = os.path.join(self.base_path, "Annotation/annotation.csv")

    def check_paths(self):
        """Check existence of all required data paths"""

        paths = {
            "video-Lie": self.video_false,
            "video-Truth": self.video_true,
            "text-Lie": self.text_false,
            "text-Truth": self.text_true,
            "audio-Lie": self.audio_false,
            "audio-Truth": self.audio_true,
            "annotation_file": self.annotation_file
        }

        print("Checking data paths:")
        all_exist = True
        for name, path in paths.items():
            exists = os.path.exists(path)
            status = "✅" if exists else "❌"
            print(f"   {status} {name}: {path}")
            if not exists:
                all_exist = False
                # If the path does not exist, try listing the contents of the parent directory to aid in debugging.
                parent_dir = os.path.dirname(path)
                if os.path.exists(parent_dir):
                    files = os.listdir(parent_dir)
                    print(f"      Parent directory content: {files}")

        return all_exist


if __name__ == "__main__":
    config = DataConfig()
    config.check_paths()
