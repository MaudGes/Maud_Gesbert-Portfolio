from flask import Flask, render_template, request, abort
import pandas as pd
import os
import pickle

app = Flask(__name__)

@app.route("/", endpoint="home")
def home():
    lang = request.args.get("lang", "fr")
    if lang == "en":
        return render_template("index_en.html")
    return render_template("index.html")

@app.route("/cv", endpoint="cv")
def cv():
    lang = request.args.get("lang", "fr")
    if lang == "en":
        return render_template("cv/cv_en.html")
    return render_template("cv/cv.html")

@app.route("/projects/heart_disease", methods=["GET", "POST"])
def heart_disease():
    prediction = None
    lang = request.args.get("lang") or request.form.get("lang") or "fr"

    if request.method == "POST":
        try:
            # Récupérer les valeurs du formulaire
            features = [
                int(request.form["GenHlth"]),
                int(request.form["Age"]),
                int(request.form["Stroke"]),
                int(request.form["Sex"]),
                int(request.form["HighChol"]),
                int(request.form["HighBP"]),
                int(request.form["Diabetes"]),
                float(request.form["PhysHlth"]),
                float(request.form["BMI"]),
                int(request.form["DiffWalk"]),
            ]

            columns = [
                "GenHlth", "Age", "Stroke", "Sex", "HighChol",
                "HighBP", "Diabetes", "PhysHlth", "BMI", "DiffWalk"
            ]

            input_data = pd.DataFrame([features], columns=columns)

            model_path = os.path.join("templates", "projects", "heart_disease", "heart_disease_cart_pipeline.pkl")
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            prediction_result = model.predict(input_data)[0]
            if lang == "en":
                prediction = (
                    "✅ Risk of heart disease detected"
                    if prediction_result == 1
                    else "✅ No heart disease risk detected"
                )
            else:
                prediction = (
                    "✅ Risque de maladie cardiaque détecté"
                    if prediction_result == 1
                    else "✅ Aucun risque de maladie cardiaque détecté"
                )

        except Exception as e:
            prediction = f"❌ Error during prediction: {e}" if lang == "en" else f"❌ Erreur lors de la prédiction : {e}"

    # Choisir le bon fichier HTML selon la langue
    if lang == "en":
        return render_template("projects/heart_disease/heart_disease_en.html", prediction=prediction)
    return render_template("projects/heart_disease/heart_disease.html", prediction=prediction)

@app.route("/projects/aloe", endpoint="aloe_vera")
def aloe_vera():
    lang = request.args.get("lang", "fr")
    if lang == "en":
        return render_template("projects/aloe/aloe_en.html")
    return render_template("projects/aloe/aloe.html")


@app.route("/projects/<project>")
def show_project(project):
    if project == "heart_disease":
        return heart_disease()  # redirige vers la fonction spéciale
    try:
        return render_template(f"projects/{project}/{project}.html")
    except:
        abort(404)

if __name__ == "__main__":
    app.run(debug=True)
