# 📩 SMS Spam Detection System

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![NLP](https://img.shields.io/badge/NLP-Text%20Processing-purple)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Status](https://img.shields.io/badge/Project-Completed-success)

A Python-based Natural Language Processing and Machine Learning project that classifies SMS messages as Spam or Ham using TF-IDF and the Multinomial Naive Bayes algorithm.

---

## 📌 Project Overview

SMS spam messages can contain unwanted advertisements, fraudulent offers, suspicious links, and misleading information.

This project uses Natural Language Processing (NLP) and Machine Learning to analyze SMS messages and classify them into two categories:

- **Spam:** Unwanted or potentially suspicious messages.
- **Ham:** Normal and legitimate messages.

The system learns patterns from labelled messages and uses them to classify new SMS messages.

## 🎯 Objectives

- Load and validate SMS datasets.
- Clean and preprocess text messages.
- Convert text into numerical features using TF-IDF.
- Train a spam classification model.
- Evaluate classification performance.
- Predict labels for new SMS messages.
- Export prediction results into CSV files.

## ✨ Features

- **Dataset Validation:** Checks missing values, invalid labels, and empty messages.
- **Text Preprocessing:** Converts text into lowercase and removes unnecessary whitespace.
- **TF-IDF Feature Extraction:** Converts text messages into weighted numerical features.
- **Machine Learning:** Uses Multinomial Naive Bayes for classification.
- **Model Evaluation:** Calculates accuracy, precision, recall, and F1-score.
- **Confusion Matrix:** Displays correct and incorrect classifications.
- **SMS Prediction:** Classifies new messages as spam or ham.
- **CSV Export:** Saves prediction results for further analysis.

## ⚙️ Project Workflow

1. Load SMS dataset.
2. Validate dataset columns and labels.
3. Preprocess text messages.
4. Convert messages into numerical features using TF-IDF.
5. Split the dataset into training and testing sets.
6. Train the Multinomial Naive Bayes model.
7. Evaluate model performance.
8. Predict new SMS messages.
9. Export prediction results.

## 🧠 Concepts Used

- Python Functions and Modular Programming
- Natural Language Processing (NLP)
- Regular Expressions
- Text Preprocessing
- TF-IDF Vectorization
- Machine Learning Classification
- Multinomial Naive Bayes
- Train-Test Split
- Model Evaluation
- Confusion Matrix
- Precision, Recall and F1-score
- CSV File Handling
- Exception Handling

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Pandas | Dataset handling |
| NumPy | Numerical operations |
| Scikit-learn | Machine Learning and NLP |
| TF-IDF | Text feature extraction |
| Multinomial Naive Bayes | Spam classification |

## 📂 Project Structure

```text
sms-spam-detector/
│
├── main.py
├── analyzer.py
├── spam_data.csv
├── requirements.txt
├── .gitignore
└── README.md
```

### File Description

- `main.py` – Controls the complete application workflow.
- `analyzer.py` – Contains data validation, preprocessing, TF-IDF extraction, model training, evaluation, prediction, and export functions.
- `spam_data.csv` – Sample labelled SMS dataset.
- `requirements.txt` – Required Python libraries.
- `.gitignore` – Excludes unnecessary generated files.
- `README.md` – Project documentation.

## 🔍 NLP and Machine Learning Model

### Text Preprocessing

Messages are converted to lowercase and unnecessary whitespace is removed.

### TF-IDF Vectorization

TF-IDF converts text messages into numerical feature vectors.

It uses:
- Term Frequency (TF)
- Inverse Document Frequency (IDF)

### Classification Algorithm

**Multinomial Naive Bayes**

The model learns word patterns from labelled messages and predicts whether new messages belong to the spam or ham category.

### Dataset Split

- Training Data: 80%
- Testing Data: 20%

Stratified splitting is used to preserve the class distribution across training and testing datasets.

## 📊 Model Evaluation

The project uses the following evaluation metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures overall correct predictions |
| Precision | Measures correctness of predicted spam messages |
| Recall | Measures how many actual spam messages are detected |
| F1-score | Balances precision and recall |
| Confusion Matrix | Displays classification results and errors |

## 📁 Generated Output

The project generates:

**`spam_predictions.csv`**

It contains:
- Original SMS message
- Predicted label (spam or ham)

## 🚀 Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/ayushman-ece/sms-spam-detector.git
```

### 2. Navigate to the project folder

```bash
cd sms-spam-detector
```

### 3. Create a virtual environment (optional)

```bash
python -m venv .venv
```

Activate it:

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the project

```bash
python main.py
```

## 📦 Requirements

```text
pandas
numpy
scikit-learn
```

## 📚 Learning Outcomes

Through this project, I learned:

- How Natural Language Processing works.
- How to clean and preprocess text data.
- How TF-IDF converts text into numerical features.
- How Naive Bayes performs text classification.
- How to split data into training and testing sets.
- How to evaluate classification models.
- How to make predictions on new messages.
- How to organize an NLP project using modular programming.

## ⚠️ Limitations

- Uses a small synthetic dataset containing only 30 messages.
- Predictions may not generalize to real-world SMS messages.
- Does not support multilingual spam detection.
- Does not detect malicious URLs independently.
- Uses basic text preprocessing.
- Does not perform real-time SMS monitoring.

## 🔮 Future Improvements

- Integrate a larger real-world SMS spam dataset.
- Improve model performance using cross-validation.
- Compare Naive Bayes with Logistic Regression and other classifiers.
- Add multilingual message classification.
- Develop an interactive web interface.
- Add suspicious URL detection.
- Build a real-time spam detection application.

## 👨‍💻 Author

**Ayushman Tiwari**

GitHub: [@ayushman-ece](https://github.com/ayushman-ece)

---

⭐ If you find this project useful, consider giving the repository a star!

**Project #86 — SMS Spam Detection System**