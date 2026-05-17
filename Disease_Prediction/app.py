from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(
open(
"models/model.pkl",
"rb"
)
)

@app.route(
"/",
methods=["GET","POST"]
)

def home():

    prediction=""

    if request.method=="POST":

        age=float(
        request.form["age"]
        )

        sex=float(
        request.form["sex"]
        )

        chest=float(
        request.form["chest_pain"]
        )

        bp=float(
        request.form["bp"]
        )

        chol=float(
        request.form["cholesterol"]
        )

        fbs=float(
        request.form["fbs"]
        )

        ecg=float(
        request.form["ecg"]
        )

        maxhr=float(
        request.form["max_hr"]
        )

        angina=float(
        request.form["angina"]
        )

        oldpeak=float(
        request.form["oldpeak"]
        )

        slope=float(
        request.form["slope"]
        )

        features=np.array([[

        age,
        sex,
        chest,
        bp,
        chol,
        fbs,
        ecg,
        maxhr,
        angina,
        oldpeak,
        slope

        ]])

        result=model.predict(
        features
        )

        prob=model.predict_proba(
        features
        )

        if result[0]==1:

            prediction=f"""
⚠ High Heart Disease Risk

Risk Probability:
{prob[0][1]*100:.2f}%
"""

        else:

            prediction=f"""
✅ Low Heart Disease Risk

Probability:
{prob[0][0]*100:.2f}%
"""

    return render_template(
    "index.html",
    prediction=prediction
    )

if __name__=="__main__":

    app.run(
    debug=True
    )