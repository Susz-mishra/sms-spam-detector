# 📱 SMS Spam Prediction using Machine Learning

A machine learning based **SMS spam detection system** that classifies text messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP), TF-IDF feature extraction, and supervised machine learning algorithms.

## 📌 Project Overview

Spam messages are unwanted and potentially harmful messages that can contain misleading offers, advertisements, or fraudulent content.

This project builds a text classification system that analyzes the content of an SMS message and predicts whether it is **Spam** or **Ham**.

The project follows an end-to-end NLP and machine learning workflow:

```text
SMS Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Train-Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Spam / Ham Prediction
```

## 🎯 Objectives

* Analyze an SMS text dataset.
* Clean and prepare the dataset.
* Perform exploratory data analysis.
* Preprocess textual messages.
* Convert text into numerical features using TF-IDF.
* Train machine learning classification models.
* Compare model performance.
* Predict whether new messages are Spam or Ham.
* Save the trained model and vectorizer for future use.

## 📊 Dataset

The project uses an SMS spam dataset containing **5,572 messages**.

The original dataset contains the following relevant columns:

* `v1` → Message label
* `v2` → SMS message

The columns are renamed to:

```text
label
message
```

The two classes are:

* **Ham** — legitimate/non-spam message
* **Spam** — unwanted/spam message

## 🔍 Exploratory Data Analysis

The project performs exploratory analysis to understand the distribution and characteristics of SMS messages.

EDA includes:

* Spam vs Ham class distribution
* Spam/Ham percentage visualization
* Message length analysis
* Word count analysis
* Distribution of message lengths by class

Additional features are created for analysis:

```python
message_length
word_count
```

## 🧹 Text Preprocessing

The SMS messages are cleaned before converting them into machine learning features.

The preprocessing pipeline includes:

* Converting text to lowercase
* Removing numbers
* Removing punctuation
* Removing English stopwords
* Creating a cleaned message column

Example:

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(
        [word for word in text.split() if word not in stop_words]
    )
    return text
```

## 🔢 TF-IDF Vectorization

After preprocessing, the cleaned SMS messages are converted into numerical feature vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The project uses:

```python
TfidfVectorizer(max_features=3000)
```

The dataset is divided into:

```text
Training Data: 80%
Testing Data: 20%
```

with:

```python
random_state=42
```

The resulting training TF-IDF matrix contains:

```text
4457 samples × 3000 features
```

## 🤖 Machine Learning Models

Two supervised machine learning algorithms are implemented and compared.

### 1. Logistic Regression

Logistic Regression is used as a linear classification model for distinguishing between Spam and Ham messages.

### 2. Multinomial Naive Bayes

Multinomial Naive Bayes is particularly suitable for text classification tasks involving word-frequency based features such as TF-IDF.

## 📈 Model Performance

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

### Model Comparison

| Model                   |   Accuracy | Spam Precision | Spam Recall | Spam F1-Score |
| ----------------------- | ---------: | -------------: | ----------: | ------------: |
| Logistic Regression     |     95.07% |            96% |         66% |           78% |
| Multinomial Naive Bayes | **97.22%** |       **100%** |     **79%** |       **88%** |

### Multinomial Naive Bayes

The model achieved:

```text
Accuracy: 97.22%

Spam:
Precision: 1.00
Recall:    0.79
F1-Score:  0.88
```

Confusion Matrix:

```text
[[965   0]
 [ 31 119]]
```

This means that in the test set:

* 965 Ham messages were correctly classified.
* 119 Spam messages were correctly classified.
* 31 Spam messages were classified as Ham.
* 0 Ham messages were classified as Spam.

### Logistic Regression

The model achieved:

```text
Accuracy: 95.07%

Spam:
Precision: 0.96
Recall:    0.66
F1-Score:  0.78
```

Confusion Matrix:

```text
[[961   4]
 [ 51  99]]
```

## 🧪 Sample Predictions

The trained system was tested on sample SMS messages.

| Message                                            | Prediction |
| -------------------------------------------------- | ---------- |
| "WINNER!! You have won a free iPhone. Call now!!!" | Spam       |
| "Hey, are we still meeting for lunch today?"       | Ham        |
| "URGENT! Claim your free vacation now."            | Spam       |
| "Don't forget to bring your homework."             | Ham        |

## 💾 Model Persistence

The trained model and TF-IDF vectorizer are saved using `joblib`.

```python
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
```

Generated files:

```text
model.pkl
vectorizer.pkl
```

These can be loaded later to make predictions on new SMS messages without retraining the model.

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* NLTK
* Scikit-learn
* Joblib
* Jupyter Notebook

## 📁 Project Structure

```text
sms-spam-prediction/
│
├── SMS_SPAM_prediction.ipynb
├── spam.csv
├── model.pkl
├── vectorizer.pkl
├── README.md
├── requirements.txt
└── .gitignore
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project directory

```bash
cd sms-spam-prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the notebook

```bash
jupyter notebook
```

Open:

```text
SMS_SPAM_prediction.ipynb
```

and run the cells sequentially.

## 📚 Key Learning Outcomes

Through this project, I practiced:

* Natural Language Processing fundamentals
* Text cleaning and preprocessing
* Stopword removal
* Exploratory data analysis
* TF-IDF feature extraction
* Train-test splitting
* Text classification
* Logistic Regression
* Multinomial Naive Bayes
* Confusion matrix analysis
* Precision, recall and F1-score
* Model persistence using Joblib

## ⚠️ Limitations

The current implementation can be improved further by:

* Handling class imbalance more explicitly
* Testing on larger and more diverse datasets
* Experimenting with additional NLP techniques
* Hyperparameter tuning
* Trying additional machine learning models
* Evaluating performance on real-world incoming messages


## 👨‍💻 Author

**Anuj kumar Mishra**

Computer Science Student | AI/ML & NLP Enthusiast
