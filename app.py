from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model/liver_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([np.array(features)])
    result = 'Positive for Cirrhosis' if prediction[0] == 1 else 'Negative for Cirrhosis'
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)