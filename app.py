from flask import Flask, render_template, abort

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/cv")
def cv():
    return render_template("cv/cv.html")

@app.route("/projects/<project>")
def show_project(project):
    try:
        return render_template(f"projects/{project}/{project}.html")
    except:
        abort(404)

if __name__ == "__main__":
    app.run(debug=True)
