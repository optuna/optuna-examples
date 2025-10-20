# SVM TF-IDF Sentiment (TweetEval) tuned with Optuna

This example tunes a TF-IDF + LinearSVC pipeline on the TweetEval **sentiment** dataset (3 classes: negative, neutral, positive) using Optuna.

**Run:**
```bash
python examples/sklearn/svm_tfidf_tweeteval_sentiment.py --n-trials 20 --max-train 20000


