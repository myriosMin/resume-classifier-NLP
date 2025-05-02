# 🧠 Resume Classifier and Ranker using TF-IDF and BERT

This project explores both **traditional NLP techniques** and **transformer-based models** to solve two problems:

1. **Cosine Similarity Ranking**: Rank resumes by similarity to a target resume (e.g., mine)
2. **Resume Classification**: Classify resumes into job categories using a fine-tuned BERT model

Built as an individual assignment for the NLP module in Year 2 of NYP’s DADE program.

---

## 💾 Dataset

* Source: Resume dataset from assignment portal
* Rows before preprocessing: **35,447**
* Rows after dropping duplicates: **8,278** *(Yes, our teacher is evil.)*
* Each resume has text, category label, and additional metadata

---

## 🧹 Preprocessing Pipeline

```text
Raw CSV
  ↓  (drop duplicates, 77% rows removed)
Lowercase + Remove special chars
  ↓
Acronym-aware filtering (e.g., keep SAP, ERP)
  ↓
Tokenization & POS-aware Lemmatization
  ↓
Stopword removal
  ↓
Save clean text to CSV (for both TF-IDF and BERT)
```

---

## 📌 PART 1: Resume Ranking using Vectorization + Cosine Similarity

### Vectors Implemented:

| Vectors     | Conclusion                             |
| ----------- | -------------------------------------- |
| TF-IDF      | Best performance & interpretability    |
| Word2Vec    | Dense embeddings, good structure       |
| PPMI        | Captures co-occurrence nicely          |
| BERT Embeds | Heavy weight, captures complex relations, hallucination |

### 🎯 Ranking Result

* TF-IDF performed **best in terms of human interpretability and separation**
* Word2Vec and BERT had denser but harder-to-interpret rankings
* PPMI lacked nuance for resume-specific semantics

---

## 🤖 PART 2: Resume Classification using Fine-Tuned BERT

### 📋 Preprocessing for BERT

* Each resume is tokenized and segmented to avoid BERT's 512-token limit
* Input sequences truncated to 128 tokens for consistency
* Manual chunking done where resumes exceed token limit
* Cleaned, stopword-free text helps speed up fine-tuning and reduces noise

### ⚙️ Model: `bert-base-uncased`

* Token length: 128
* Early layers frozen initially, then progressively unfrozen
* Fine-tuned with mixed precision (AMP) + cosine LR scheduler

### 📊 Performance (Before Optimization)

| Metric                | Value   |
| --------------------- | ------- |
| Overall Accuracy      | 84.82%  |
| Overall Loss          | 0.8906  |
| MCC Score (overall)   | 84.66%  |
| Validation Accuracy   | 88.89%  |
| Validation Loss       | 0.0245  |
| Training Time (Total) | 337.66s |
| Inference Speed       | 0.0199s |

> Final MCC score on test set from peer data (Jin Bin): **81.5**, Test Accuracy: **81.7%** — indicates generalization is strong, even across similar corpora.

## 🔁 Optimization

* Unfroze last 50 → then last 100 layers gradually
* Applied ReduceLROnPlateau, EarlyStopping, and checkpointing
* Used lower learning rate (1e-5) for later fine-tuning
* Quantized model post-training to reduce size by \~75%

### 📊 BERT Model Comparison (Before vs After Optimization)

| Metric                    | Fine-Tuned BERT | Optimized BERT    |
| ------------------------- | --------------- | ----------------- |
| Training Time (Total)     | 337.66s         | **67.81s**        |
| Training Time per Epoch   | 9.13s           | **1.44s**         |
| Validation Accuracy       | 88.89%          | 88.89%            |
| Validation Loss           | 0.0245          | 0.0245            |
| Overall Accuracy          | 84.82%          | 84.82%            |
| Overall Loss              | 0.8906          | 0.8906            |
| MCC Score (Overall)       | 84.66%          | 84.66%            |
| Inference Time per Sample | 0.0199s         | **0.0191s**       |
| Model Size                | Full            | **\~75% smaller** |

> Optimized BERT retains classification performance while being significantly faster and lighter — better suited for deployment.

| Metric            | Value                             |
| ----------------- | --------------------------------- |
| Training Accuracy | 97.57%                            |
| Validation Acc    | 94.13%                            |
| MCC Score         | 81.5 (test)                       |
| Inference Speed   | 0.0074s/image                     |
| Model Size        | Reduced by \~75% via quantization |

> Despite using fewer tokens and pre-cleaned text, BERT performs well after careful tuning and becomes efficient for deployment.

---

## 🔍 Summary Comparison

| Task           | Winner     | Why                                             |
| -------------- | ---------- | ----------------------------------------------- |
| Ranking        | **TF-IDF** | Light, interpretable, top human-aligned results |
| Classification | **BERT**   | Deep contextual understanding, high accuracy    |

---

## 👤 Author

\[Year 2, NLP Assignment, Diploma in AI & Data Engineering, Nanyang Polytechnic]
**[Min Phyo Thura](https://github.com/myriosMin)**

---

Thanks for reading! This project proves both classical and modern NLP approaches have their own strengths depending on context, compute, and task goals. Feel free to connect with me if you have any doubts or want to collaborate.