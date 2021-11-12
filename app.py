from flask import Flask, render_template, request
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open('Kidney_Disease_Prediction.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        specificgravity = float(request.form['specific gravity'])
        hypertension = float(request.form['hypertension'])
        haemoglobin = float(request.form['haemoglobin'])
        diabetesmellitus = float(request.form['diabetes mellitus'])
        albumin = float(request.form['albumin'])
        appetite = float(request.form['appetite'])
        serumcreatinine = float(request.form['serum creatinine'])
        puscell = float(request.form['pus cell'])

        values = np.array([[specificgravity,hypertension, haemoglobin, diabetesmellitus, albumin, appetite, serumcreatinine, puscell]])
        prediction = model.predict(values)

        return render_template('result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)