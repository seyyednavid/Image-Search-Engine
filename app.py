import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from engine import search_image

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)

from flask import send_from_directory

# Serve files from /data
@app.route('/data/<path:filename>')
def data_files(filename):
    return send_from_directory("data", filename)

# Serve files from /uploads
@app.route('/uploads/<path:filename>')
def uploaded_files(filename):
    return send_from_directory("uploads", filename)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["query"]
        if not file:
            return render_template("index.html", error="No file uploaded")

        filename = secure_filename(file.filename)
        path = os.path.join(UPLOAD_DIR, filename)
        file.save(path)

        results = search_image(path, top_k=8)
        return render_template("results.html",
                               query_path=path,
                               results=results)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
