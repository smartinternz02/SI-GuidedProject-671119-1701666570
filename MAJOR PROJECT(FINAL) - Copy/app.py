import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='Templates')
model = pickle.load(open(r'model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=["POST"])
def submit():
    # Retrieve data from the form
    month = float(request.form['Month'])
    day = float(request.form['Day'])
    avg_temp = float(request.form['avg_temp'])
    avg_humd = float(request.form['avg_humd'])

    # Prepare data for prediction
    input_feature = np.array([[month, day, avg_temp, avg_humd]])
    columns = ['Month', 'Day', 'avg_temp', 'avg_humd']
    data = pd.DataFrame(input_feature, columns=columns)

    # Make prediction
    prediction = model.predict(data)
    out = prediction[0]

    return render_template("index.html", result=out)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=1111, debug=True)
