# Detecting Coordinated Information Operations on Twitter via Text Analysis

This project addresses the growing challenge of identifying coordinated information operations (IO) on social media platforms, particularly Twitter. By combining linguistic features from tweets with behavioral patterns of users, we build predictive models to detect potential manipulation in online discourse.

-	面向社交平台中疑似协同操控账号的检测任务，解决“短文本 + 异质用户行为”双模态数据下的标签稀缺与特征碎片化问题 （见models）
-	实现文本清洗、关键词掩码、子词编码等完整预处理流程，解决社交媒体口语化噪声与拼写变异问题，提升模型泛化能力（见utils-preprocess.py)

## Data Sources

- **IO tweets**: [Kaggle - Russian Troll Tweets](https://www.kaggle.com/datasets/vikasg/russian-troll-tweets)
- **Non-IO tweets**: [Kaggle - US Election Tweets](https://www.kaggle.com/datasets/matt0922/us-presidential-election-tweets)

## Models
-	使用 PyTorch 与 Hugging face Transformers 构建多模型架构：包括文本微调（M1）、MLP 分类器（M2）、行为特征随机森林（M3）及融合模型（M4），提升跨模态信号识别能力
We evaluated four models (见notebooks）：

| Model | Description |
|-------|-------------|
| M1 | Fine-tuned BERTweet |
| M2 | BERTweet embeddings + MLP (Selected) |
| M3 | Random Forest on user behavioral features |
| M4 | Stacked model (BERTweet + projected user features via MLP) |

> **Final selected model**: M2 — best trade-off between performance and stability, with F1 = 0.843 on test data. 在高类别不平衡场景中，通过精度-召回权衡优化指标，最终模型 F1 达 0.84+

## How to Run

```bash
pip install -r requirements.txt
python inference.py --input tweet_data.csv --output prediction.csv
