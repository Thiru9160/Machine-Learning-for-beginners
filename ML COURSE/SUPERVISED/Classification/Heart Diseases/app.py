from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    # Pass empty dictionary to avoid undefined error in template
    return render_template('index.html', previous_values={})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['age']),
            int(request.form['sex']),
            int(request.form['chestpaintype']),
            float(request.form['restingbp']),
            float(request.form['cholesterol']),
            int(request.form['fastingbs']),
            int(request.form['restingecg']),
            float(request.form['maxhr']),
            int(request.form['exerciseangina']),
            float(request.form['oldpeak']),
            int(request.form['slope']),
            int(request.form['ca']),
            int(request.form['thalassemia'])
        ]
        input_array = np.array([features])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            result = "The patient is likely to have heart disease."
        else:
            result = "The patient is unlikely to have heart disease."
    except Exception as e:
        result = f"Error in input values: {e}"
    return render_template('index.html', prediction_text=result, previous_values=request.form)

if __name__ == '__main__':
    app.run(debug=True)
