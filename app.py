from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv
from threading import Lock
from datetime import datetime
from PyPDF2 import PdfReader
import os, io, traceback

load_dotenv()

ALLOWED_EXTENSIONS = {"pdf", "doc", "docx"}
TEN_MB = 10 * 1024 * 1024
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "2"))
PORT = int(os.environ.get("PORT", "5050"))

app = Flask(__name__, instance_relative_config=True)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", os.urandom(24)),
    UPLOAD_FOLDER=os.environ.get("UPLOAD_FOLDER", os.path.join(app.instance_path, "uploads")),
)
Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

_jobs = {}
_lock = Lock()
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


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

def _preprocess_pdf(job_id, filepath):
    _set_job(job_id, status="RUNNING", started_at=datetime.utcnow().isoformat() + "Z")
    try:
        with open(filepath, "rb") as f:
            reader = PdfReader(f)
            info = reader.metadata or {}
            pages = len(reader.pages)
        meta = {str(k): str(v) for k, v in info.items()} if info else {}
        result = {"file": os.path.basename(filepath), "pages": pages, "metadata": meta}
        _set_job(job_id, status="SUCCESS", finished_at=datetime.utcnow().isoformat() + "Z", result=result)
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

if __name__ == "__main__":
    app.run(debug=True, port=PORT)