from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os

from flask import Flask

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")

# serve upload page with same template just to match nav behavior
@app.route("/upload")
def upload_page():
    return render_template("index.html")

@app.route("/upload", methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({"message": "No image provided"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"message": "No image selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"message": "File type not allowed"}), 400
    
    # Save the file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    return jsonify({"message": f"Image saved as {filename}"})

@app.route("/search")
def search_page():
    return render_template("search.html")

@app.route("/timeline")
def timeline_page():
    return render_template("timeline.html")

if __name__ == "__main__":
    app.run(debug=True)