import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import re


class TextFeatureExtractor:
    """Text feature extractor using mBERT (multilingual BERT)"""

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.model = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.model.eval()  # Set to evaluation mode

    def extract_text_features(self, text_content):
        """Extract mBERT-based features from text"""
        if not text_content or not text_content.strip():
            return np.zeros(768)  # mBERT 输出维度是 768

        try:
            # Clean and preprocess text
            text_content = re.sub(r'\s+', ' ', text_content).strip()

            # Tokenize input text
            inputs = self.tokenizer(
                text_content,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True
            )

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Use the [CLS] token's embedding as sentence representation
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

            return cls_embedding  # Shape: (768,)
        except Exception as e:
            print(f"Error extracting mBERT features: {e}")
            return np.zeros(768)
