from flask import Flask, render_template, request, abort
import pickle
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/cv")
def cv():
    return render_template("cv/cv.html")

@app.route("/projects/heart_disease", methods=["GET", "POST"])
def heart_disease():
    prediction = None

    if request.method == "POST":
        try:
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

            # Charger le modèle
            model_path = os.path.join("templates", "projects", "heart_disease", "heart_disease_cart_pipeline.pkl")
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            prediction_result = model.predict([features])[0]
            prediction = (
                "✅ Risque de maladie cardiaque détecté"
                if prediction_result == 1
                else "✅ Aucun risque de maladie cardiaque détecté"
            )

        except Exception as e:
            prediction = f"❌ Erreur lors de la prédiction : {e}"

    return render_template("projects/heart_disease/heart_disease.html", prediction=prediction)

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
