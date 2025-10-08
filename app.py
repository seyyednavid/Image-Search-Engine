import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from engine import search_image


UPLOAD_DIR = "uploads"
DATA_DIR = "data"
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


@app.route("/", methods=["GET"])
def shop():
    """Show shoe shop with all images from data/."""
    images = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    return render_template("shop.html", images=images)


@app.route("/search/<filename>", methods=["GET"])
def search(filename):
    """Search similar shoes given a chosen image from data folder."""
    query_path = os.path.join(DATA_DIR, filename)
    results = search_image(query_path, top_k=6)
    return render_template("results.html",
                           query_path=f"data/{filename}",
                           results=results)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)