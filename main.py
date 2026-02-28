from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime
from pathlib import Path
import pytesseract
from PIL import Image
from google import genai
from dotenv import load_dotenv

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
    api_key = os.getenv('AIzaSyD6iMnwYrPsBJNr9GShpfWBlwv-qsFQ8vA', '').strip()
    if not api_key:
        if not ENV_FILE.exists():
            raise ValueError('GENAI_API_KEY is not configured. Missing .env file. Create it with: cp .env.example .env')
        raise ValueError('GENAI_API_KEY is not configured in .env')

    model = os.getenv('GENAI_MODEL', 'gemini-2.0-flash').strip() or 'gemini-2.0-flash'
    fallback_models_raw = os.getenv('GENAI_MODEL_FALLBACKS', 'gemini-1.5-flash')
    fallback_models = [m.strip() for m in fallback_models_raw.split(',') if m.strip()]
    candidate_models = []
    for candidate in [model, *fallback_models]:
        if candidate not in candidate_models:
            candidate_models.append(candidate)

    client = genai.Client(api_key=api_key)
    last_permission_error = None

    for candidate_model in candidate_models:
        try:
            prompt = (
                'You summarize OCR text from historical images. '
                'Return a concise summary in 3-5 bullet points.\n\n'
                f'OCR text:\n{text}'
            )
            response = client.models.generate_content(
                model=candidate_model,
                contents=prompt,
            )
            summary = (getattr(response, 'text', None) or '').strip()
            if not summary:
                raise ValueError('No summary returned by model')
            return summary, candidate_model
        except Exception as e:
            error_text = str(e).lower()
            if '403' in error_text or 'permission' in error_text or 'forbidden' in error_text:
                last_permission_error = e
                continue
            raise

    tried_models = ', '.join(candidate_models)
    if last_permission_error:
        raise PermissionError(f'No accessible GenAI model. Tried: {tried_models}. Last error: {last_permission_error}')
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
    except ValueError as e:
        return jsonify({'message': str(e)}), 500
    except Exception as e:
        error_text = str(e).lower()
        if '401' in error_text or 'invalid api key' in error_text or 'authentication' in error_text or 'unauthenticated' in error_text:
            return jsonify({'message': 'GenAI authentication failed. Check GENAI_API_KEY.'}), 401
        if '403' in error_text or 'permission' in error_text or 'access denied' in error_text or 'forbidden' in error_text:
            return jsonify({'message': 'GenAI access denied for this model/project. Try a different model or check project permissions.'}), 403
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