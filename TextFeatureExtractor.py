import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import TextBlob

# ======================== 文本特征提取器 ========================
class TextFeatureExtractor:
    """文本特征提取器"""

    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        self.fitted = False

    def extract_text_features(self, text_content):
        """提取文本特征"""
        if not text_content.strip():
            return np.zeros(110)  # 100 TF-IDF + 10 统计特征

        try:
            # 1. 基本统计特征
            word_count = len(text_content.split())
            char_count = len(text_content)
            sentence_count = len(re.split(r'[.!?]+', text_content))
            avg_word_length = np.mean([len(word) for word in text_content.split()]) if word_count > 0 else 0

            # 2. 情感特征
            blob = TextBlob(text_content)
            sentiment_polarity = blob.sentiment.polarity
            sentiment_subjectivity = blob.sentiment.subjectivity

            # 3. 语言复杂度特征
            unique_words = len(set(text_content.lower().split()))
            lexical_diversity = unique_words / word_count if word_count > 0 else 0

            # 4. 停用词和标点符号比例
            punctuation_count = sum(1 for char in text_content if char in '.,!?;:')
            punctuation_ratio = punctuation_count / char_count if char_count > 0 else 0

            uppercase_ratio = sum(1 for char in text_content if char.isupper()) / char_count if char_count > 0 else 0

            # 统计特征向量
            stats_features = [
                word_count, char_count, sentence_count, avg_word_length,
                sentiment_polarity, sentiment_subjectivity, lexical_diversity,
                punctuation_ratio, uppercase_ratio, unique_words
            ]

            return np.array(stats_features)

        except Exception as e:
            print(f"提取文本特征时出错: {e}")
            return np.zeros(110)

    def fit_tfidf(self, texts):
        """训练TF-IDF模型"""
        valid_texts = [text for text in texts if text.strip()]
        if valid_texts:
            self.tfidf.fit(valid_texts)
            self.fitted = True

    def get_tfidf_features(self, text):
        """获取TF-IDF特征"""
        if not self.fitted or not text.strip():
            return np.zeros(100)
        try:
            return self.tfidf.transform([text]).toarray()[0]
        except:
            return np.zeros(100)