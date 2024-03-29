# Flipkart Sentiment Analysis
## Introduction
The Flipkart Sentiment Analysis project is designed to analyze customer reviews on Flipkart and classify them as positive or negative using natural language processing (NLP) techniques and machine learning algorithms. The project utilizes AWS ECS (Elastic Container Service) for deployment, Flask for building the web application, and NLTK (Natural Language Toolkit) and Scikit-learn for data preprocessing and model building.

## Tech Stack
* **AWS ECS**: Used for deployment to provide scalability, reliability, and ease of management.
* **Flask**: Used for building the web application to interact with users and display results.
* **NLTK**: Used for text preprocessing tasks such as tokenization, stemming, and lemmatization.
* **Scikit-learn**: Used for building and training machine learning models for sentiment analysis.

## Workflow
* **Data Collection**: Customer reviews from Flipkart are collected and stored for analysis.
* **Data Preprocessing**: The collected data is preprocessed using NLTK for tasks like tokenization, removing stopwords, and lemmatization.
* **Feature Extraction**: Text data is converted into numerical features using techniques like TF-IDF (Term Frequency-Inverse Document Frequency).
* **Model Building**: Machine learning models such as Logistic Regression or Random Forest are trained on the preprocessed data.
* **Deployment**: The trained model is deployed on AWS ECS using Flask to create a web application accessible to users.
* **User Interaction****: Users can input their text reviews into the web application, and the model predicts the sentiment (positive or negative) of the input text.
* **Display Results**: The predicted sentiment is displayed to the user via the web interface.

## Conclusion
The Flipkart Sentiment Analysis project demonstrates the application of NLP and machine learning techniques to analyze customer sentiments from textual data. By deploying the model on AWS ECS and building a user-friendly web interface with Flask, the project provides a scalable and accessible solution for analyzing customer feedback on Flipkart.

Join me in exploring customer sentiments on Flipkart with this exciting project!
* AWS link: http://13.51.168.53:5000/
