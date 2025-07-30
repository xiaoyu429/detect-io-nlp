# utils/preprocess.py
"""clean_tweet: 清洗文本，去除 URL、@、非 ASCII、空格。
mask_keywords: 根据高频主题词列表进行精确屏蔽。
preprocess_tweet: 组合 pipeline，主函数。
传入 keyword_list，如 ["trump", "maga", "hillary", ...]。
后续加入BPE分词或tokenizer处理。
"""

import re
import string

def clean_tweet(text):
    """
    Standard text cleaning:
    - Lowercase
    - Replace URLs and mentions
    - Remove non-ascii characters
    - Remove excess whitespace
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "URL", text)  # Replace URLs
    text = re.sub(r"@\w+", "@user", text)          # Replace mentions
    text = re.sub(r"[^ -~]", "", text)             # Remove non-ASCII
    text = re.sub(r"\s+", " ", text).strip()
    return text

def mask_keywords(text, keyword_list, mask_token="[MASK]"):
    """
    Replace known topic-specific keywords with a mask token.
    Case-insensitive exact match masking.
    """
    for word in keyword_list:
        pattern = r"\b" + re.escape(word.lower()) + r"\b"
        text = re.sub(pattern, mask_token, text)
    return text

def preprocess_tweet(text, keyword_list):
    """
    Full preprocessing pipeline:
    cleaning + keyword masking
    """
    cleaned = clean_tweet(text)
    masked = mask_keywords(cleaned, keyword_list)
    return masked
