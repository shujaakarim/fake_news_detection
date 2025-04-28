from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load model and vectorizer
with open('fake_news_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Allow letters, numbers, and spaces
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


# Create Flask app
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ''
    news_text = ''
    error_message = ''

    if request.method == 'POST':
        news_text = request.form['news']

        # Server-side validation
        if not news_text:  # Invalid input check for empty input
            error_message = "Please enter some news text."
        else:
            cleaned_text = clean_text(news_text)
            vectorized_text = vectorizer.transform([cleaned_text])
            result = model.predict(vectorized_text)[0]
            prediction = "Real News" if result == 1 else "Fake News"

    return render_template('index.html', prediction=prediction, news_text=news_text, error_message=error_message)


if __name__ == '__main__':
    app.run(debug=True)
