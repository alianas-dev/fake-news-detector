# 🔍 Fake News Detector

A machine learning web app that classifies news articles as **Real** or **Fake** using Natural Language Processing.

Built with Python, scikit-learn, and Streamlit.

🚀 **[Live Demo →](https://alianas-dev-fake-news-detector.streamlit.app)**

---

## How It Works

1. Input news text is cleaned — lowercased, URLs removed, stopwords filtered, lemmatized
2. Text is converted to numerical features using **TF-IDF** (unigrams + bigrams)
3. A **Random Forest classifier** predicts the label with a confidence score

---

## Models Compared

| Vectorizer | Best Model | CV Accuracy |
|---|---|---|
| TF-IDF | Random Forest | ~99% |
| Bag of Words | Random Forest | ~99% |
| Binary BoW | Random Forest | ~98% |
| Word2Vec | Logistic Regression | ~92% |

TF-IDF + Random Forest was selected as the final model based on cross-validation performance.

---

## Stack

- Python · scikit-learn · NLTK · Gensim
- Streamlit (frontend)
- Random Forest · Logistic Regression · SVM · Naive Bayes · Decision Tree

---

## Run Locally

```bash
git clone https://github.com/alianas-dev/fake-news-detector
cd fake-news-detector
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚠️ Accuracy Disclaimer

The model shows ~99% accuracy on the test set — but that number comes with important context:

- **Small dataset** — trained on only 1,680 labeled articles. Real-world fake news detectors use millions of samples
- **Domain-specific** — dataset is mostly political and general English news. May misclassify sports, science, or entertainment articles
- **Clean data bias** — training data was well-structured. Real-world news is messier
- **No context awareness** — the model reads words, not meaning. A well-written fake article can fool it
- **Not production-ready** — built as a university project and portfolio demo, not a tool for real fact-checking

Use it as a demonstration of NLP and ML concepts, not as a reliable fact-checker.

---

## Dataset

- 1,680 labeled news articles (`fake` / `Real`)
- Preprocessed: stopword removal, lemmatization, outlier filtering
- 80/20 train-test split with stratification

---

## Author

**Mohammad Anas (Ali Anas)**  
BS Artificial Intelligence — KIET, Karachi  
[LinkedIn](https://www.linkedin.com/in/mohammad-anas-501595335) · [GitHub](https://github.com/alianas-dev)
