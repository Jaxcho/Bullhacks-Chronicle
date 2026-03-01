from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
from pathlib import Path
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from urllib import request as urlrequest, error as urlerror

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path=ENV_FILE, override=True)

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


def generate_summary_from_text(text):
    model = os.getenv('OLLAMA_MODEL', 'llama3.1').strip() or 'llama3.1'
    fallback_models_raw = os.getenv('OLLAMA_MODEL_FALLBACKS', 'qwen2.5:7b,phi3:mini')
    fallback_models = [m.strip() for m in fallback_models_raw.split(',') if m.strip()]
    candidate_models = []
    for candidate in [model, *fallback_models]:
        if candidate not in candidate_models:
            candidate_models.append(candidate)

    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').strip().rstrip('/')
    endpoint = f'{base_url}/api/generate'
    prompt = (
        'You summarize OCR text from historical images. '
        'Return a concise summary in 3-5 bullet points.\n\n'
        f'OCR text:\n{text}'
    )
    last_model_error = None

    for candidate_model in candidate_models:
        try:
            payload = json.dumps({
                'model': candidate_model,
                'prompt': prompt,
                'stream': False
            }).encode('utf-8')
            req = urlrequest.Request(
                endpoint,
                data=payload,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urlrequest.urlopen(req, timeout=120) as response:
                body = response.read().decode('utf-8', errors='replace')
            result = json.loads(body) if body else {}
            summary = (result.get('response') or '').strip()
            if not summary:
                raise ValueError('No summary returned by model')
            return summary, candidate_model
        except urlerror.HTTPError as e:
            error_body = e.read().decode('utf-8', errors='replace') if hasattr(e, 'read') else ''
            error_text = error_body.lower()
            if e.code == 404 or 'not found' in error_text:
                last_model_error = e
                continue
            raise ValueError(f'Ollama request failed ({e.code}): {error_body or e.reason}')
        except urlerror.URLError as e:
            raise ConnectionError(
                f'Cannot connect to Ollama at {base_url}. Start it with: ollama serve'
            ) from e

    tried_models = ', '.join(candidate_models)
    if last_model_error:
        raise ValueError(
            f'No available Ollama model. Tried: {tried_models}. Pull one first, e.g.: ollama pull {model}'
        )
    raise ValueError(f'No model available to generate summary. Tried: {tried_models}')


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
    # optional date field (YYYY-MM-DD)
    date_str = request.form.get('date', '').strip()

    # Extract text from image using OCR
    ocr_text = ""
    try:
        img = Image.open(filepath)
        ocr_text = pytesseract.image_to_string(img)
    except Exception as e:
        print(f"OCR failed for {filename}: {e}")
        ocr_text = ""

    ai_summary = ""
    ai_summary_model = ""
    if ocr_text.strip():
        try:
            ai_summary, ai_summary_model = generate_summary_from_text(ocr_text)
        except Exception as e:
            print(f"Summary generation failed for {filename}: {e}")

    metadata = load_metadata()
    entry = {
        'title': title,
        'tags': tags,
        'comments': [],
        'extracted_text': ocr_text,
        'uploaded_at': datetime.utcnow().isoformat() + 'Z'
    }
    if ai_summary:
        entry['summary'] = ai_summary
        entry['summary_model'] = ai_summary_model
        entry['summary_generated_at'] = datetime.utcnow().isoformat() + 'Z'
    if date_str:
        entry['date'] = date_str
    metadata[filename] = entry
    save_metadata(metadata)

    return jsonify({
        "message": f"Image saved as {filename}",
        "filename": filename,
        "summary_generated": bool(ai_summary)
    })


@app.route("/search")
def search_page():
    q = request.args.get('q', '').strip().lower()
    start = request.args.get('start_date', '').strip()
    end = request.args.get('end_date', '').strip()
    metadata = load_metadata()
    items = []
    for fn, entry in metadata.items():
        e = dict(entry)
        e['filename'] = fn
        items.append(e)
    # sort by uploaded_at desc
    try:
        items.sort(key=lambda x: x.get('uploaded_at', ''), reverse=True)
    except Exception:
        pass
    # filter by tag
    if q:
        items = [img for img in items if any(q in t.lower() for t in img.get('tags', []))]
    # filter by date range if provided (use entry['date'] or uploaded_at)
    def in_range(img):
        if not (start or end):
            return True
        d = img.get('date') or img.get('uploaded_at', '')[:10]
        try:
            if start and d < start:
                return False
            if end and d > end:
                return False
        except Exception:
            pass
        return True
    items = [img for img in items if in_range(img)]
    return render_template("search.html", images=items, query=q, start_date=start, end_date=end)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/timeline")
def timeline_page():
    # prepare timeline entries sorted by date or upload
    metadata = load_metadata()
    items = []
    for fn, entry in metadata.items():
        e = dict(entry)
        e['filename'] = fn
        # use 'date' if present else uploaded_at
        d = e.get('date') or e.get('uploaded_at', '')[:10]
        e['_sort_date'] = d
        items.append(e)
    # sort chronologically ascending
    try:
        items.sort(key=lambda x: x.get('_sort_date', ''))
    except Exception:
        pass
    return render_template("timeline.html", items=items)


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


@app.route('/comment/delete', methods=['POST'])
def delete_comment():
    data = request.get_json(force=True, silent=True)
    if not data or 'filename' not in data or 'index' not in data:
        return jsonify({'message': 'filename and index required'}), 400

    filename = data['filename']
    index = data['index']

    metadata = load_metadata()
    if filename not in metadata:
        return jsonify({'message': 'file not found in metadata'}), 404

    comments = metadata[filename].get('comments', [])
    if index < 0 or index >= len(comments):
        return jsonify({'message': 'invalid comment index'}), 400

    comments.pop(index)
    save_metadata(metadata)

    return jsonify({'message': 'comment deleted'})


# detail page for an image
@app.route('/image/<filename>')
def image_detail(filename):
    metadata = load_metadata()
    if filename not in metadata:
        return "Not found", 404
    entry = metadata[filename]
    return render_template('image.html', filename=filename, entry=entry)


@app.route('/image/<filename>/summarize', methods=['POST'])
def summarize_image(filename):
    metadata = load_metadata()
    if filename not in metadata:
        return jsonify({'message': 'file not found in metadata'}), 404

    extracted_text = (metadata[filename].get('extracted_text') or '').strip()
    if not extracted_text:
        return jsonify({'message': 'no OCR text available to summarize'}), 400

    try:
        summary, used_model = generate_summary_from_text(extracted_text)
    except PermissionError as e:
        return jsonify({'message': str(e)}), 403
    except ConnectionError as e:
        return jsonify({'message': str(e)}), 503
    except ValueError as e:
        return jsonify({'message': str(e)}), 500
    except Exception as e:
        error_text = str(e).lower()
        if 'cannot connect to ollama' in error_text or 'connection refused' in error_text:
            return jsonify({'message': 'Cannot connect to Ollama. Start it with: ollama serve'}), 503
        if 'no available ollama model' in error_text or 'pull one first' in error_text:
            return jsonify({'message': str(e)}), 500
        return jsonify({'message': f'Failed to generate summary: {e}'}), 500

    metadata[filename]['summary'] = summary
    metadata[filename]['summary_model'] = used_model
    metadata[filename]['summary_generated_at'] = datetime.utcnow().isoformat() + 'Z'
    save_metadata(metadata)

    return jsonify({'message': 'summary generated', 'summary': summary})


# deletion
@app.route('/image/<filename>/delete', methods=['POST'])
def delete_image(filename):
    metadata = load_metadata()
    if filename in metadata:
        del metadata[filename]
        save_metadata(metadata)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
    return jsonify({'message': 'deleted'})






if __name__ == "__main__":
    port = int(os.getenv('PORT', '5000'))
    host = os.getenv('HOST', '127.0.0.1')
    app.run(debug=True, host=host, port=port)