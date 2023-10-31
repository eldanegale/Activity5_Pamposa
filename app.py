import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('model/svcmodel.pkl', 'rb'))

feature_names = ['age', 'experience', 'income', 'family', 'CCAvg', 'education', 'mortgage', 'personal_loan', 'securities_account', 'cd_account', 'online']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = [float(request.form.get(field, 0)) for field in feature_names]
    prediction = model.predict([user_input])[0]
    # Interpret the prediction result
    result = 'Eligible for Credit Card' if prediction == 1 else 'Not Eligible for Credit Card'
    return render_template('index.html', prediction_result=result)

if __name__ == '__main__':
    app.run(debug=True)
