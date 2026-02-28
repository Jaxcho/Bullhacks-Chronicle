from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
METADATA_FILE = os.path.join(UPLOAD_FOLDER, 'metadata.json')

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_metadata():
    if not os.path.exists(METADATA_FILE):
        return {}
    try:
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_metadata(data):
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@app.route("/")
def home():
    return render_template("index.html")


# serve upload page (same page used for uploads)
@app.route("/upload")
def upload_page():
    return render_template("upload.html")


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

    # Build metadata
    title = request.form.get('title', '').strip()
    tags_raw = request.form.get('tags', '')
    tags = [t.strip() for t in tags_raw.split(',') if t.strip()] if tags_raw else []

    metadata = load_metadata()
    metadata[filename] = {
        'title': title,
        'tags': tags,
        'comments': [] ,
        'uploaded_at': datetime.utcnow().isoformat() + 'Z'
    }
    save_metadata(metadata)

    return jsonify({"message": f"Image saved as {filename}", "filename": filename})


@app.route("/search")
def search_page():
    return render_template("search.html")


@app.route("/timeline")
def timeline_page():
    return render_template("timeline.html")


@app.route('/comment', methods=['POST'])
def add_comment():
    data = request.get_json(force=True, silent=True)
    if not data or 'filename' not in data or 'comment' not in data:
        return jsonify({'message': 'filename and comment required'}), 400

    filename = data['filename']
    comment_text = data['comment'].strip()
    if not comment_text:
        return jsonify({'message': 'comment is empty'}), 400

    metadata = load_metadata()
    if filename not in metadata:
        return jsonify({'message': 'file not found in metadata'}), 404

    comment_entry = {
        'text': comment_text,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    metadata[filename].setdefault('comments', []).append(comment_entry)
    save_metadata(metadata)

    return jsonify({'message': 'comment added', 'comment': comment_entry})


@app.route('/generate_tags', methods=['POST'])
def generate_tags():
    # Accepts an image file in 'image' and calls make_tags.make_tags(image)
    if 'image' not in request.files:
        return jsonify({'message': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'message': 'No image selected'}), 400

    # Read bytes
    image_bytes = file.read()

    try:
        # Import user-provided tag generator; user should implement make_tags.make_tags
        import make_tags

        # Try calling with bytes first; if the user's function expects a path, fall back
        try:
            tags = make_tags.make_tags(image_bytes)
        except TypeError:
            # save a temporary file and pass path
            tmp_name = os.path.join(app.config['UPLOAD_FOLDER'], '_tmp_' + secure_filename(file.filename))
            with open(tmp_name, 'wb') as tmpf:
                tmpf.write(image_bytes)
            try:
                tags = make_tags.make_tags(tmp_name)
            finally:
                try:
                    os.remove(tmp_name)
                except Exception:
                    pass

        # Normalize tags into a list of strings
        if isinstance(tags, str):
            tags_list = [t.strip() for t in tags.split(',') if t.strip()]
        elif isinstance(tags, (list, tuple)):
            tags_list = [str(t).strip() for t in tags]
        else:
            tags_list = []

        return jsonify({'tags': tags_list})
    except Exception as e:
        return jsonify({'message': 'error generating tags', 'error': str(e)}), 500


@app.route('/save_tags', methods=['POST'])
def save_tags():
    data = request.get_json(force=True, silent=True)
    if not data or 'filename' not in data or 'tags' not in data:
        return jsonify({'message': 'filename and tags required'}), 400

    filename = data['filename']
    tags_in = data['tags']
    tags = [str(t).strip() for t in tags_in if t and str(t).strip()]
    if not tags:
        return jsonify({'message': 'no tags provided'}), 400

    metadata = load_metadata()
    if filename not in metadata:
        return jsonify({'message': 'file not found'}), 404

    existing = metadata[filename].get('tags', [])
    # merge preserving order and dedupe
    merged = []
    for t in existing + tags:
        if t not in merged:
            merged.append(t)
    metadata[filename]['tags'] = merged
    save_metadata(metadata)

    return jsonify({'message': 'tags saved', 'tags': merged})


if __name__ == "__main__":
    app.run(debug=True)