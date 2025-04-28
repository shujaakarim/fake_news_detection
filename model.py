# Basic libraries
import pandas as pd
import numpy as np

# Text processing libraries
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Machine Learning libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import pickle

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Set path
data_path = r'E:\Noor_Khan\COMPUTER SCIENCE JOURNEY\Internships\Remot ( DevelopersHub )\Projects\fake_news_detection'

# Load datasets
fake_df = pd.read_csv(data_path + r'\Fake.csv')
true_df = pd.read_csv(data_path + r'\True.csv')

# Add labels
fake_df['label'] = 0
true_df['label'] = 1

# Merge datasets
news_df = pd.concat([fake_df, true_df]).reset_index(drop=True)
news_df = news_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle


# Check basic info
print(news_df.info())

# Check missing values
print(news_df.isnull().sum())

# Plot class distribution
sns.countplot(x='label', data=news_df)
plt.title('Class Distribution (0: Fake, 1: Real)')
plt.show()


# Initialize
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Clean text function
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special characters
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

# Apply cleaning on "text" column
news_df['text'] = news_df['text'].apply(clean_text)


# Features and target
X = news_df['text']
y = news_df['label']

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)


# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": MultinomialNB()
}

# Train and evaluate
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"Accuracy for {model_name}: {accuracy_score(y_test, preds)}")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\n" + "-"*50 + "\n")


# Save the model
best_model = LogisticRegression(max_iter=1000)
best_model.fit(X_train, y_train)

with open('fake_news_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

# Save the vectorizer too
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)


# Predict function
def predict_news(news_text):
    with open('fake_news_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)

    news_text = clean_text(news_text)
    vectorized_text = vectorizer.transform([news_text])
    prediction = model.predict(vectorized_text)

    return "Real News" if prediction[0] == 1 else "Fake News"

# Example
print(predict_news(" shujaat is pm of pakistan "))

