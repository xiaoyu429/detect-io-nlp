# Detecting Coordinated Information Operations on Twitter via Text Analysis

This project addresses the growing challenge of identifying coordinated information operations (IO) on social media platforms, particularly Twitter. By combining linguistic features from tweets with behavioral patterns of users, we build predictive models to detect potential manipulation in online discourse.

## Data Sources

- **IO tweets**: [Kaggle - Russian Troll Tweets](https://www.kaggle.com/datasets/vikasg/russian-troll-tweets)
- **Non-IO tweets**: [Kaggle - US Election Tweets](https://www.kaggle.com/datasets/matt0922/us-presidential-election-tweets)

## Models

We evaluated four models:

| Model | Description |
|-------|-------------|
| M1 | Fine-tuned BERTweet |
| M2 | BERTweet embeddings + MLP (Selected) |
| M3 | Random Forest on user behavioral features |
| M4 | Stacked model (BERTweet + projected user features via MLP) |

> **Final selected model**: M2 — best trade-off between performance and stability, with F1 = 0.843 on test data.

## How to Run

```bash
pip install -r requirements.txt
python inference.py --input tweet_data.csv --output prediction.csv
