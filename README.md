# **Task-2:** Amazon Customer Review Sentiment Analysis

**COMPANY:** CODTECH IT SOLUTIONS

**NAME:** RAVI YADAV

**INTERN ID:** CT08IIJ

**DOMAIN:** MACHINE LEARNING

**DURATION:** 4 WEEKS

**MENTOR:** NEELA SANTOSH


# **Description**

## **Project Overview**  
The **Amazon Customer Review Sentiment Analysis** project aims to classify customer reviews as positive or negative based on sentiment analysis techniques. Customer reviews are crucial for businesses to understand user satisfaction, improve products, and enhance customer service. This project utilizes **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques to analyze customer sentiments from text reviews.  

Using a dataset containing **10,252** reviews, we preprocess the text data and apply **TF-IDF vectorization** to extract meaningful features. The target labels (sentiments) are encoded using **label encoding**, and a **Logistic Regression** model is trained to predict the sentiment of unseen reviews. Finally, we evaluate the model’s performance using various metrics to determine its effectiveness.  

---

## **Dataset Information**  
The dataset consists of **10,252** customer reviews from Amazon, each labeled as either **positive** or **negative** based on sentiment. The reviews are processed and prepared for machine learning using vectorization and encoding techniques.  

---

## **Data Preprocessing and Feature Engineering**  

### **1. TF-IDF Vectorization**  
- We apply **Term Frequency-Inverse Document Frequency (TF-IDF)** transformation on both **training** and **test** data.  
- TF-IDF helps convert textual data into numerical format while reducing the impact of frequently occurring words (e.g., "the", "is", "and").  
- This technique improves the model’s ability to focus on important words that contribute to sentiment classification.  

### **2. Label Encoding**  
- Since the target labels are categorical (positive/negative), we use **label encoding** to convert them into numerical format:  
  - **Positive sentiment → 2**  
  - **Negative sentiment → 0**
  - **Neutral sentiment → 1**  
- This step ensures compatibility with machine learning algorithms.  

---

## **Model Implementation: Logistic Regression**  
We use **Logistic Regression**, a widely used linear classifier for binary classification problems.  

### **Why Logistic Regression?**  
✅ Efficient for text classification tasks  
✅ Provides probabilistic predictions (useful for confidence estimation)  
✅ Works well with TF-IDF transformed data  
✅ Interpretable and less prone to overfitting compared to complex models  

---

## **Model Training and Evaluation**  
After preprocessing, we split the data into training and testing sets and train the **Logistic Regression** model. We then evaluate its performance using various metrics:  

### **Performance Metrics:**  
📌 **Accuracy** – Measures the percentage of correct predictions.  
📌 **Precision** – Determines how many predicted positive sentiments are actually positive.  
📌 **Recall** – Measures the ability to identify actual positive sentiments.  
📌 **F1-score** – A balance between precision and recall.  
📌 **Confusion Matrix** – Provides insight into correct and incorrect classifications.  

---

## **Expected Outcomes**  
✅ The model will classify customer reviews as **positive** or **negative** with high accuracy.  
✅ The use of **TF-IDF** ensures that meaningful words contribute to sentiment classification.  
✅ The **Logistic Regression** model provides a simple yet effective approach for text-based sentiment analysis.  
✅ Performance metrics will validate the effectiveness of our approach.  

---

## **Conclusion**  
This project demonstrates a practical application of **NLP** and **machine learning** in sentiment analysis. Businesses can use such models to analyze customer feedback, improve products, and enhance customer satisfaction. The use of **TF-IDF vectorization**, **label encoding**, and **Logistic Regression** ensures an efficient and interpretable sentiment classification system. Future improvements may include experimenting with **deep learning models** or **hyperparameter tuning** for better accuracy.  
