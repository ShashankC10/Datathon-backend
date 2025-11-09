from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv
from threading import Lock
from datetime import datetime
from PyPDF2 import PdfReader
import os, io, traceback
import json

# <-- NEW: import your classifier class
from Inference import ContextualPDFClassifier

load_dotenv()

ALLOWED_EXTENSIONS = {"pdf", "doc", "docx"}
TEN_MB = 10 * 1024 * 1024
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
PORT = int(os.environ.get("PORT", "5050"))

app = Flask(__name__, instance_relative_config=True)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", os.urandom(24)),
    UPLOAD_FOLDER=os.environ.get("UPLOAD_FOLDER", os.path.join(app.instance_path, "uploads")),
)
Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

# <-- NEW: results folder for annotated PDFs / JSON
RESULTS_FOLDER = os.path.join(app.instance_path, "results")
Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)
SETTINGS_PATH = Path(app.instance_path) / "prompt_settings.json"
SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
DEFAULTS = ContextualPDFClassifier.DEFAULT_PROMPT_LIBRARY


_jobs = {}
_lock = Lock()
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# <-- NEW: single classifier instance
_classifier = ContextualPDFClassifier()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def file_size_bytes_and_reset(f):
    stream = f.stream
    try:
        if hasattr(stream, "seek") and hasattr(stream, "tell"):
            pos = stream.tell()
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(pos, os.SEEK_SET)
            return size
    except Exception:
        pass
    data = stream.read()
    size = len(data)
    f.stream = io.BytesIO(data)
    f.stream.seek(0)
    return size


def _set_job(job_id, **fields):
    with _lock:
        _jobs[job_id].update(fields)


def _snapshot_jobs():
    with _lock:
        jobs = list(_jobs.values())
    return sorted(jobs, key=lambda j: j.get("created_at", ""), reverse=True)


# ====== REPLACED: preprocess now uses ContextualPDFClassifier ======

def _preprocess_pdf(job_id, filepath):
    """Run contextual classification on the given PDF and store outputs.
    Resulting PDF will keep the same base filename and be stored under RESULTS_FOLDER.
    """
    _set_job(job_id, status="RUNNING", started_at=datetime.utcnow().isoformat() + "Z")
    try:
        # Compute simple PDF stats (pages) for dashboard, independent of classifier
        pages = None
        try:
            with open(filepath, "rb") as f:
                reader = PdfReader(f)
                pages = len(reader.pages)
        except Exception:
            pages = None

        base_pdf_name = os.path.basename(filepath)  # keep same name
        result_pdf_path = os.path.join(RESULTS_FOLDER, base_pdf_name)
        result_json_path = os.path.splitext(result_pdf_path)[0] + ".json"

        prompt_overrides = load_settings()
        # Run the classifier
        classification = _classifier.classify_pdf_contextually(
            pdf_path=filepath,
            output_json=result_json_path,
            output_pdf=result_pdf_path,
            prompt_overrides=prompt_overrides,
        )

        final_categories = classification.get("final_classification", {}).get("final_categories", [])

        _set_job(
            job_id,
            status="SUCCESS",
            finished_at=datetime.utcnow().isoformat() + "Z",
            result={
                "pages": pages,
                "classified_pdf": os.path.basename(result_pdf_path),
                "json_result": os.path.basename(result_json_path),
                "categories": final_categories,
            },
        )
    except Exception:
        _set_job(job_id, status="FAILED", finished_at=datetime.utcnow().isoformat() + "Z", error=traceback.format_exc())


@app.get("/")
def index():
    return render_template("upload.html")


@app.post("/upload")
def upload_many():
    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        flash("No files selected")
        return redirect(url_for("index"))

    valid, total_size = [], 0
    for f in files:
        if not allowed_file(f.filename):
            flash(f"Only PDF/DOC/DOCX files allowed: '{f.filename}' rejected")
            return redirect(url_for("index"))
        size = file_size_bytes_and_reset(f)
        total_size += size
        valid.append((f, size))

    count = len(valid)
    limit = count * TEN_MB
    if total_size > limit:
        flash(f"Total upload size exceeds {count} × 10 MB limit.")
        return redirect(url_for("index"))

    created_jobs = []
    for f, _ in valid:
        job_id = uuid4().hex
        filename = secure_filename(f.filename)
        stored = f"{job_id}_{filename}"
        dest = os.path.join(app.config["UPLOAD_FOLDER"], stored)
        f.save(dest)
        with _lock:
            _jobs[job_id] = {
                "id": job_id,
                "status": "PENDING",
                "created_at": datetime.utcnow().isoformat() + "Z",
                "stored_file": stored,
                "original_file": filename,
            }
        _executor.submit(_preprocess_pdf, job_id, dest)
        created_jobs.append({"job_id": job_id, "filename": filename})

    return render_template("success.html", files=created_jobs)


@app.get("/jobs")
def jobs_page():
    jobs = _snapshot_jobs()
    return render_template("jobs.html", jobs=jobs)


@app.get("/api/jobs")
def api_list_jobs():
    return jsonify(_snapshot_jobs())


@app.get("/api/jobs/<job_id>")
def api_get_job(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404
    return jsonify(job)


@app.get("/api/jobs/<job_id>/result")
def api_get_result(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404
    if job.get("status") != "SUCCESS":
        return jsonify({"error": f"Job not complete (status={job.get('status')})"}), 409
    return jsonify(job["result"])


# <-- NEW: serve result PDFs / JSONs
@app.get("/results/<filename>")
def serve_result(filename):
    return send_from_directory(RESULTS_FOLDER, filename)


@app.post("/api/jobs/<job_id>/rerun")
def api_rerun_job(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404
    if job["status"] not in {"FAILED", "SUCCESS"}:
        return jsonify({"error": "Job not complete"}), 409

    src_path = os.path.join(app.config["UPLOAD_FOLDER"], job["stored_file"])
    if not os.path.exists(src_path):
        return jsonify({"error": "Original file missing"}), 404

    new_id = uuid4().hex
    with _lock:
        _jobs[new_id] = {
            "id": new_id,
            "status": "PENDING",
            "created_at": datetime.utcnow().isoformat() + "Z",
            "stored_file": job["stored_file"],
            "original_file": job["original_file"],
        }

    _executor.submit(_preprocess_pdf, new_id, src_path)
    return jsonify({"message": "Rerun started", "new_job_id": new_id})

def load_settings():
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            # ensure keys exist
            return {
                "categories": data.get("categories", DEFAULTS["categories"]),
                "rules": data.get("rules", DEFAULTS["rules"]),
                "contextual_logic": data.get("contextual_logic", DEFAULTS["contextual_logic"]),
            }
        except Exception:
            pass
    return {k: list(v) for k, v in DEFAULTS.items()}

def save_settings(d):
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "categories": d.get("categories", []),
            "rules": d.get("rules", []),
            "contextual_logic": d.get("contextual_logic", []),
        }, f, indent=2, ensure_ascii=False)

# --- page to edit settings ---
@app.get("/settings")
def settings_page():
    return render_template(
        "settings.html",
        settings=load_settings(),
        defaults=DEFAULTS,
    )

# --- read current settings (optional) ---
@app.get("/api/settings")
def api_get_settings():
    return jsonify(load_settings())

# --- save settings ---
@app.post("/api/settings")
def api_save_settings():
    try:
        data = request.get_json(force=True) or {}
        # coerce to lists of strings
        def cleanse(x):
            if not isinstance(x, list): return []
            return [str(i).strip() for i in x if str(i).strip()]
        clean = {
            "categories": cleanse(data.get("categories")),
            "rules": cleanse(data.get("rules")),
            "contextual_logic": cleanse(data.get("contextual_logic")),
        }
        # basic guards
        if not clean["categories"]:
            return jsonify({"error": "categories cannot be empty"}), 400
        save_settings(clean)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=PORT)