# Sentiment Classification & Fake Review Detection on Amazon Reviews

## 📌 Project Overview
This project aims to develop a Natural Language Processing (NLP) system to analyze Amazon customer reviews. The system performs two main tasks:
1.  **Sentiment Classification:** Classifying reviews as Positive, Neutral, or Negative based on the text and score.
2.  **Fake Review Detection:** Identifying suspicious or fake reviews using heuristic-based labeling (since ground-truth labels are not available).

## 🚀 Features
-   **Data Analysis:** Exploratory Data Analysis (EDA) on Amazon reviews.
-   **Text Preprocessing:** Cleaning, tokenization, stopword removal, and lemmatization.
-   **Feature Extraction:** Using TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numerical vectors.
-   **Fake Review Labeling:** Heuristic-based approach to generate "Fake" labels based on duplicates, short length, and unhelpfulness.
-   **Model Implementation:**
    -   **Machine Learning:** Naive Bayes, Logistic Regression, Random Forest.
    -   **Deep Learning:** LSTM, CNN (Convolutional Neural Networks).
    -   **Transformers:** BERT / RoBERTa.

## 📂 Dataset
The project uses the `amazon_reviews.csv` dataset, which contains the following key columns:
-   `Id`, `ProductId`, `UserId`, `ProfileName`: Identifiers for the review, product, and user.
-   `HelpfulnessNumerator`, `HelpfulnessDenominator`: Metrics for review helpfulness.
-   `Score`: Rating given by the user (converted to sentiment labels).
-   `Time`: Timestamp of the review.
-   `Summary`: Brief summary of the review.
-   `Text`: The full review content.

## 🛠️ Tech Stack
-   **Language:** Python
-   **Libraries:**
    -   `pandas`, `numpy` (Data Manipulation)
    -   `matplotlib`, `seaborn` (Visualization)
    -   `nltk`, `re` (Text Processing)
    -   `scikit-learn` (Machine Learning & Metrics)
    -   `tensorflow` / `keras` / `pytorch` (Deep Learning & Transformers)

## 🔧 Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```
2.  Install the required dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn nltk scikit-learn tensorflow
    ```
    *(Note: Adjust requirements based on specific versions used in the notebook)*

## 📖 Usage
1.  Ensure `amazon_reviews.csv` is in the project directory.
2.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook "Sentiment_Classification_&_Fake_Review_Detection_on_Amazon_Reviews.ipynb"
    ```
3.  Run the cells sequentially to perform data loading, preprocessing, training, and evaluation.

## 📊 Methodology
### 1. Label Generation (Fake Reviews)
Since real "fake" labels are missing, we generate them based on:
-   **Duplicates:** Exact text matches found elsewhere.
-   **Length:** Reviews with fewer than 5 words.
-   **Unhelpfulness:** Reviews with high vote counts but extremely low helpfulness ratios.

### 2. Preprocessing Pipeline
-   Lowercase conversion.
-   Removal of HTML tags, URLs, and special characters.
-   Tokenization and Stopword removal.
-   Lemmatization using WordNet.

### 3. Modeling
The project compares traditional ML algorithms (Naive Bayes, Random Forest) with advanced Deep Learning models (LSTM, BERT) to find the best performing approach for both sentiment classification and fake review detection.

## 📈 Results
The models are evaluated using metrics such as:
-   Accuracy
-   Precision, Recall, F1-Score
-   ROC-AUC Curve
