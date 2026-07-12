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

The model achieves ~99% accuracy on the ISOT test set. A few things to keep in mind:

- **Domain-specific** — trained on political/world news (2016–2018). May struggle with sports, science, or entertainment articles
- **Time-bound** — news patterns from 2016–2018 may differ from today's misinformation styles
- **English only** — performs poorly on non-English text
- **Not a fact-checker** — the model detects writing patterns, not actual facts. A well-written fake article can still fool it
- Use for portfolio/demo purposes only

---

## Dataset

- **ISOT Fake News Dataset** — 44,898 labeled news articles
- Source: University of Victoria, Canada
- Two files: `True.csv` (real news) + `Fake.csv` (fake news)
- Covers political and world news from 2016–2018
- 80/20 train-test split with stratification

---

## Author

**Mohammad Anas (Ali Anas)**  
BS Artificial Intelligence — KIET, Karachi  
[LinkedIn](https://www.linkedin.com/in/mohammad-anas-501595335) · [GitHub](https://github.com/alianas-dev)
