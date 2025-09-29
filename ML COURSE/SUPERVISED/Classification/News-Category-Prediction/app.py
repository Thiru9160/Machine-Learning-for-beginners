from flask import Flask, request, render_template, redirect, url_for
import joblib
import re
import string

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('modell.pkl')
vectorizer = joblib.load('vectorizer.pkl')
#link between web page and trained 
# Store predictions in session
history = []

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text

@app.route('/', methods=['GET', 'POST'])
def home():
    global history
    prediction_text = ""
    if request.method == 'POST':
        headline = request.form['headline']
        cleaned = clean_text(headline)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]
        prediction_text = f'Predicted Category: {prediction}'
        history.insert(0, {'headline': headline, 'prediction': prediction})
        if len(history) > 5:
            history.pop()
    return render_template('index.html', prediction_text=prediction_text, history=history)

if __name__ == "__main__":
    app.run(debug=True)
