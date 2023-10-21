# Restaurant Food Reviews Classification using NLP with TF-IDF Transformer

This project aims to classify restaurant food reviews into "liked" or "not liked" categories using Natural Language Processing (NLP) techniques and the TF-IDF Transformer. The project also employs several machine learning models, including Multinomial Naive Bayes, CatBoost, and XGBoost, for the classification task. Additionally, a web application using Streamlit has been developed to provide an interactive interface for users to input restaurant reviews and receive predictions on whether the food is liked or not.

## Project Overview

- **Objective**: To classify restaurant food reviews as "liked" or "not liked" using NLP techniques and machine learning models.
- **Tools and Technologies**: Python, Scikit-learn, TF-IDF Transformer, Multinomial Naive Bayes, CatBoost, XGBoost, Streamlit.
- **Dataset**: A labeled dataset of restaurant food reviews containing text data and corresponding labels (liked/not liked).

## Project Components

### 1. Data Preprocessing

Before applying machine learning models, the project involves data preprocessing steps such as:
- Text cleaning (removing special characters, stopwords, etc.).
- Tokenization and text vectorization using TF-IDF Transformer.

### 2. Model Building and Training

The following machine learning models are implemented and trained on the preprocessed data:
- **Multinomial Naive Bayes**: A probabilistic model used for text classification tasks.
- **CatBoost**: A gradient boosting algorithm that handles categorical features effectively and provides high accuracy.
- **XGBoost**: An optimized gradient boosting algorithm with excellent predictive performance.

Each model is trained on the labeled dataset to learn the patterns in restaurant food reviews.

### 3. Model Comparison

After training, the models' performance is compared using relevant evaluation metrics such as accuracy, precision, recall, and F1-score. This comparison helps in determining which model is the most suitable for the task of restaurant food review classification and result obtained says that Multinomial Naive bayes model is more accurate 

### 4. Web Application with Streamlit

To make the classification process accessible to users, a web application is developed using Streamlit. This web app allows users to:
- Input restaurant food reviews or upload the test file.
- Click a button to obtain predictions from the best-performing model (selected during the model comparison step).
- See whether the review is classified as "liked" or "not liked."

## How to Run the Web Application

To run the web application, follow these steps:

1. Install the required libraries by running:
   ```
   pip install scikit-learn catboost xgboost streamlit
   ```

2. Clone the project repository:
   ```
   git clone <repository_url>
   ```

3. Navigate to the project directory:
   ```
   cd Restaurant_Food_Reviews_Classification
   ```

4. Run the Streamlit web application:
   ```
   streamlit run text_class.py
   ```


## Output Images

![image](https://github.com/Krishshrk/text_classification_nlp/assets/93509656/83a7bb44-3b1a-4bde-b1f2-329077d60479)

![image](https://github.com/Krishshrk/text_classification_nlp/assets/93509656/28577e42-c917-4e96-a2cd-b4d65d853d39)

![image](https://github.com/Krishshrk/text_classification_nlp/assets/93509656/31f8d108-9e58-4bd0-8d8b-b8a86579270e)


## Conclusion

This project provides a practical example of using NLP techniques and machine learning models for text classification. The web application built with Streamlit makes it user-friendly and accessible for users to classify restaurant food reviews as "liked" or "not liked." The model comparison step ensures that the best-performing model is used for predictions.
