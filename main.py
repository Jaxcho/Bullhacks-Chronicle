from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import json
import math
import re
import importlib
from itertools import combinations
from datetime import datetime
from pathlib import Path
from collections import Counter
import pytesseract
from PIL import Image, ImageOps, ImageFilter
try:
    pillow_heif = importlib.import_module('pillow_heif')
    pillow_heif.register_heif_opener()
    HEIF_ENABLED = True
except Exception:
    HEIF_ENABLED = False
from dotenv import load_dotenv
from urllib import request as urlrequest, error as urlerror

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path=ENV_FILE, override=True)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'heic', 'heif'}
METADATA_FILE = os.path.join(UPLOAD_FOLDER, 'metadata.json')

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def make_unique_filename(filename):
    name, ext = os.path.splitext(filename)
    candidate = filename
    counter = 1
    while os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], candidate)):
        candidate = f"{name}_{counter}{ext}"
        counter += 1
    return candidate


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


def tokenize_text(text):
    if not text:
        return []
    return [tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) > 2]


def entry_summary_text(entry):
    summary = (entry.get('summary') or '').strip()
    if summary:
        return summary
    return (entry.get('extracted_text') or '').strip()


def cosine_similarity(text_a, text_b):
    tokens_a = tokenize_text(text_a)
    tokens_b = tokenize_text(text_b)
    if not tokens_a or not tokens_b:
        return 0.0

    vec_a = Counter(tokens_a)
    vec_b = Counter(tokens_b)
    intersection = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[token] * vec_b[token] for token in intersection)
    norm_a = math.sqrt(sum(value * value for value in vec_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_related_entries(filename, metadata, limit=5):
    if filename not in metadata:
        return []

    target = metadata[filename]
    target_summary = entry_summary_text(target)
    if not target_summary:
        return []
    target_tags = set(tag.lower() for tag in (target.get('tags') or []) if tag)

    related = []
    for other_filename, other_entry in metadata.items():
        if other_filename == filename:
            continue

        other_summary = entry_summary_text(other_entry)
        summary_score = cosine_similarity(target_summary, other_summary)

        other_tags = set(tag.lower() for tag in (other_entry.get('tags') or []) if tag)
        union = target_tags | other_tags
        tag_score = (len(target_tags & other_tags) / len(union)) if union else 0.0

        score = (0.85 * summary_score) + (0.15 * tag_score)
        if score <= 0:
            continue

        related.append({
            'filename': other_filename,
            'title': other_entry.get('title') or other_filename,
            'date': other_entry.get('date') or (other_entry.get('uploaded_at', '')[:10]),
            'score': round(score, 4),
        })

    related.sort(key=lambda item: item['score'], reverse=True)
    return related[:limit]


def parse_entry_datetime(entry):
    date_str = (entry.get('date') or '').strip()
    if date_str:
        try:
            return datetime.strptime(date_str, '%Y-%m-%d')
        except Exception:
            pass

    uploaded_at = (entry.get('uploaded_at') or '').strip()
    if uploaded_at:
        try:
            normalized = uploaded_at.replace('Z', '+00:00')
            return datetime.fromisoformat(normalized)
        except Exception:
            pass

    return None


def generate_summary_from_text(text):
    model = os.getenv('OLLAMA_MODEL', 'gemma3:1b').strip() or 'gemma3:1b'
    fallback_models_raw = os.getenv('OLLAMA_MODEL_FALLBACKS', 'qwen2.5:7b,phi3:mini')
    fallback_models = [m.strip() for m in fallback_models_raw.split(',') if m.strip()]
    candidate_models = []
    for candidate in [model, *fallback_models]:
        if candidate not in candidate_models:
            candidate_models.append(candidate)

    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').strip().rstrip('/')
    endpoint = f'{base_url}/api/generate'
    prompt = (
        'You are summarizing a real journal entry for a timeline archive. '
        'Write a natural, human-sounding summary in 2-3 sentences. '
        'Use plain language. Avoid bullet points. '
        'Do not say phrases like "Here is a summary", "The text says", or mention OCR.\n\n'
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
            summary = re.sub(r'^(here\s*(is|\'s)\s*(a\s*)?(concise\s*)?summary\s*[:\-]?\s*)', '', summary, flags=re.IGNORECASE)
            summary = re.sub(r'^(summary\s*[:\-]\s*)', '', summary, flags=re.IGNORECASE)
            summary = re.sub(r'\s+', ' ', summary).strip()
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


def call_ollama_generate(prompt, timeout=90, model=None):
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').strip().rstrip('/')
    endpoint = f'{base_url}/api/generate'
    used_model = (model or os.getenv('OLLAMA_MODEL', 'gemma3:1b')).strip() or 'gemma3:1b'
    payload = json.dumps({
        'model': used_model,
        'prompt': prompt,
        'stream': False,
    }).encode('utf-8')
    req = urlrequest.Request(
        endpoint,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urlrequest.urlopen(req, timeout=timeout) as response:
        body = response.read().decode('utf-8', errors='replace')
    result = json.loads(body) if body else {}
    return (result.get('response') or '').strip()


def call_ollama_embedding(text):
    base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434').strip().rstrip('/')
    endpoint = f'{base_url}/api/embeddings'
    embed_model = os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text').strip() or 'nomic-embed-text'
    payload = json.dumps({
        'model': embed_model,
        'prompt': text,
    }).encode('utf-8')
    req = urlrequest.Request(
        endpoint,
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urlrequest.urlopen(req, timeout=90) as response:
        body = response.read().decode('utf-8', errors='replace')
    result = json.loads(body) if body else {}
    return result.get('embedding')


def vector_cosine_similarity(vec_a, vec_b):
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def pick_entry_text(entry):
    return (
        (entry.get('summary') or '').strip()
        or (entry.get('extracted_text') or '').strip()
        or (entry.get('title') or '').strip()
    )


def keyword_themes_from_text(text, max_items=5):
    stopwords = {
        'the', 'and', 'for', 'that', 'with', 'this', 'from', 'your', 'into', 'have', 'has', 'was', 'were',
        'is', 'are', 'but', 'not', 'you', 'its', 'their', 'them', 'about', 'there', 'then', 'than', 'they',
    }
    tokens = [tok for tok in tokenize_text(text) if tok not in stopwords]
    counts = Counter(tokens)
    return [token for token, _ in counts.most_common(max_items)]


def extract_entry_themes(text, tags):
    themes = [str(tag).strip().lower() for tag in (tags or []) if str(tag).strip()]
    if text:
        try:
            prompt = (
                'Extract 3 to 6 short theme labels from this journal entry text. '
                'Return only a comma-separated list with no extra words.\n\n'
                f'Text:\n{text}'
            )
            raw = call_ollama_generate(prompt, timeout=75)
            parsed = [segment.strip().lower() for segment in raw.split(',') if segment.strip()]
            themes.extend(parsed)
        except Exception:
            themes.extend(keyword_themes_from_text(text, max_items=5))
    unique = []
    seen = set()
    for theme in themes:
        cleaned = re.sub(r'[^a-z0-9\s\-]', '', theme).strip()
        if len(cleaned) < 2:
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)
    return unique[:8]


def build_theme_network(metadata):
    entry_nodes = {}
    theme_nodes = {}
    edges = []
    entry_vectors = {}
    entry_texts = {}
    theme_to_entries = {}
    entry_link_strength = {}
    shared_attribute_count = {}

    for filename, entry in metadata.items():
        text = pick_entry_text(entry)
        entry_texts[filename] = text
        entry_id = f'entry:{filename}'
        entry_nodes[entry_id] = {
            'id': entry_id,
            'label': entry.get('title') or filename,
            'group': 'entry',
            'shape': 'dot',
            'value': 18,
            'title': filename,
            'filename': filename,
        }
        entry_link_strength[filename] = 0.0
        shared_attribute_count[filename] = 0

        try:
            embedding = call_ollama_embedding(text) if text else None
        except Exception:
            embedding = None
        entry_vectors[filename] = embedding

        themes = extract_entry_themes(text, entry.get('tags'))
        for theme in themes:
            theme_id = f'theme:{theme}'
            if theme_id not in theme_nodes:
                theme_nodes[theme_id] = {
                    'id': theme_id,
                    'label': theme,
                    'group': 'theme',
                    'shape': 'dot',
                    'value': 10,
                }
            edges.append({
                'from': f'entry:{filename}',
                'to': theme_id,
                'value': 1,
                'width': 1.2,
                'color': {'opacity': 0.55},
            })
            theme_to_entries.setdefault(theme_id, set()).add(filename)

    similarity_threshold = float(os.getenv('NETWORK_SIM_THRESHOLD', '0.52'))
    for file_a, file_b in combinations(metadata.keys(), 2):
        vec_a = entry_vectors.get(file_a)
        vec_b = entry_vectors.get(file_b)
        if vec_a and vec_b:
            score = vector_cosine_similarity(vec_a, vec_b)
        else:
            score = cosine_similarity(entry_texts.get(file_a, ''), entry_texts.get(file_b, ''))

        is_strong = score >= similarity_threshold
        edge_width = round(0.8 + (score * 11.0), 2)
        edge_color = {'color': '#80ffdb', 'opacity': 0.75} if is_strong else {'color': '#94a3b8', 'opacity': 0.22}
        edge = {
            'from': f'entry:{file_a}',
            'to': f'entry:{file_b}',
            'value': round(max(0.05, score * 2.2), 3),
            'width': edge_width,
            'color': edge_color,
        }
        if is_strong:
            edge['label'] = f"{score:.2f}"
            edge['font'] = {'size': 10, 'color': '#9fb3ff'}
        edges.append(edge)
        entry_link_strength[file_a] += score
        entry_link_strength[file_b] += score

    # Boost entry node size when it shares themes with multiple entries and has strong mutual links.
    for theme_id, filenames in theme_to_entries.items():
        if len(filenames) < 2:
            continue
        shared_boost = min(5.0, (len(filenames) - 1) * 1.2)
        for filename in filenames:
            shared_attribute_count[filename] += 1
            entry_link_strength[filename] += shared_boost

    for filename, strength in entry_link_strength.items():
        entry_id = f'entry:{filename}'
        if entry_id not in entry_nodes:
            continue
        attribute_bonus = shared_attribute_count.get(filename, 0) * 2.4
        scaled_value = 18 + (strength * 3.6) + attribute_bonus
        entry_nodes[entry_id]['value'] = round(min(54, max(16, scaled_value)), 2)
        entry_nodes[entry_id]['has_shared_attribute'] = shared_attribute_count.get(filename, 0) > 0
        entry_nodes[entry_id]['shared_attribute_count'] = shared_attribute_count.get(filename, 0)

    nodes = list(entry_nodes.values()) + list(theme_nodes.values())
    return {'nodes': nodes, 'edges': edges}


def preprocess_image_for_ocr(image):
    processed = ImageOps.exif_transpose(image)
    if processed.mode not in ('L', 'RGB'):
        processed = processed.convert('RGB')

    width, height = processed.size
    min_side = min(width, height)
    if min_side < 1200:
        scale = 1200 / float(min_side)
        new_size = (int(width * scale), int(height * scale))
        processed = processed.resize(new_size, Image.Resampling.LANCZOS)

    gray = ImageOps.grayscale(processed)
    gray = ImageOps.autocontrast(gray)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    gray = gray.filter(ImageFilter.SHARPEN)

    binary = gray.point(lambda px: 255 if px > 145 else 0)
    return gray, binary


def clean_ocr_text(text):
    if not text:
        return ''

    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    words = [word for word in words if len(word) > 1 or word.lower() in {'a', 'i'}]
    return ' '.join(words)


def extract_text_from_image(filepath):
    with Image.open(filepath) as image:
        gray, binary = preprocess_image_for_ocr(image)

    candidates = []
    configs = [
        '--oem 3 --psm 6',
        '--oem 3 --psm 3',
        '--oem 3 --psm 11',
    ]
    for img in (gray, binary):
        for config in configs:
            try:
                text = pytesseract.image_to_string(img, config=config)
            except Exception:
                text = ''
            cleaned_text = clean_ocr_text((text or '').strip())
            if cleaned_text:
                candidates.append(cleaned_text)

    if not candidates:
        return ''
    return max(candidates, key=lambda item: (len(item.split()), len(item)))


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

    # Save the file (convert HEIC/HEIF to JPG for browser compatibility)
    original_filename = secure_filename(file.filename)
    ext = original_filename.rsplit('.', 1)[1].lower()

    if ext in {'heic', 'heif'}:
        if not HEIF_ENABLED:
            return jsonify({
                "message": "HEIC/HEIF support is not enabled. Install pillow-heif and restart the app."
            }), 400
        filename = make_unique_filename(f"{os.path.splitext(original_filename)[0]}.jpg")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            image = Image.open(file.stream)
            image.convert('RGB').save(filepath, format='JPEG', quality=95)
        except Exception as e:
            return jsonify({"message": f"Failed to convert HEIC/HEIF image: {e}"}), 400
    else:
        filename = make_unique_filename(original_filename)
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
        ocr_text = extract_text_from_image(filepath)
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
    # filter by tag/title/filename/summary
    if q:
        def matches_query(img):
            tags = [str(t).lower() for t in (img.get('tags') or [])]
            title = str(img.get('title') or '').lower()
            filename = str(img.get('filename') or '').lower()
            summary = str(img.get('summary') or '').lower()
            return (
                any(q in tag for tag in tags)
                or q in title
                or q in filename
                or q in summary
            )

        items = [img for img in items if matches_query(img)]
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
        dt = parse_entry_datetime(e)
        e['_timeline_dt'] = dt
        e['_sort_ts'] = dt.timestamp() if dt else float('inf')
        items.append(e)
    # sort chronologically ascending
    try:
        items.sort(key=lambda x: x.get('_sort_ts', float('inf')))
    except Exception:
        pass

    first_dt = None
    for item in items:
        if item.get('_timeline_dt'):
            first_dt = item['_timeline_dt']
            break

    prev_dt = None
    for item in items:
        dt = item.get('_timeline_dt')
        if dt and prev_dt:
            diff_days = max(0, (dt.date() - prev_dt.date()).days)
            gap_px = min(220, int(diff_days * 3.2))
        else:
            gap_px = 0
        item['_gap_px'] = gap_px
        prev_dt = dt if dt else prev_dt

    events = [
        {
            'date': img.get('date') or img.get('uploaded_at', '')[:10],
            'event': img.get('title') or img.get('filename'),
            'day': ((img['_timeline_dt'].date() - first_dt.date()).days if img.get('_timeline_dt') and first_dt else idx)
        }
        for idx, img in enumerate(items)
    ]
    return render_template("timeline.html", items=items, events=events)


@app.route('/network')
def network_page():
    metadata = load_metadata()
    graph = build_theme_network(metadata) if metadata else {'nodes': [], 'edges': []}
    return render_template('network.html', graph=graph)


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
    related_entries = get_related_entries(filename, metadata)
    return render_template('image.html', filename=filename, entry=entry, related_entries=related_entries)


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
    port = int(os.getenv('PORT', '5051'))
    host = os.getenv('HOST', '127.0.0.1')
    app.run(debug=True, host=host, port=port)