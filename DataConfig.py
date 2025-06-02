import os

# ======================== Data path configuration 数据路径配置 ========================
class DataConfig:
    """数据路径配置类"""
    """Data path configuration"""
    def __init__(self):
        self.base_path = os.path.join(os.path.dirname(__file__), "dataset")

        # Video path 视频路径
        self.video_false = os.path.join(self.base_path, "Clips/false")
        self.video_true = os.path.join(self.base_path, "Clips/true")

        # Text path 文本路径
        self.text_false = os.path.join(self.base_path, "Transcription/false")
        self.text_true = os.path.join(self.base_path, "Transcription/true")

        # Audio path 音频路径
        self.audio_false = os.path.join(self.base_path, "audio/false")
        self.audio_true = os.path.join(self.base_path, "audio/true")

        # Labeling files 标注文件
        self.annotation_file = os.path.join(self.base_path, "Annotation/annotation.csv")

    def check_paths(self):
        """Check all paths for existence 检查所有路径是否存在"""
        paths = {
            "video视频-Lie假话": self.video_false,
            "video视频-Truth真话": self.video_true,
            "text文本-Lie假话": self.text_false,
            "text文本-Truth真话": self.text_true,
            "audio音频-Lie假话": self.audio_false,
            "audio音频-Truth真话": self.audio_true,
            "Labeling files标注文件": self.annotation_file
        }

        print("Check data path-检查数据路径:")
        all_exist = True
        for name, path in paths.items():
            exists = os.path.exists(path)
            status = "✅" if exists else "❌"
            print(f"   {status} {name}: {path}")
            if not exists:
                all_exist = False
                # 如果路径不存在，尝试列出父目录内容帮助调试
                # If the path does not exist, try listing the contents of the parent directory to aid in debugging.
                parent_dir = os.path.dirname(path)
                if os.path.exists(parent_dir):
                    files = os.listdir(parent_dir)
                    print(f"      Parent directory content-父目录内容: {files}")

        return all_exist


if __name__ == "__main__":
    config = DataConfig()
    config.check_paths()
