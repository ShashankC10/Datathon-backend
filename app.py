from flask import Flask, render_template, request, redirect, url_for, flash, jsonify,send_from_directory
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv
from threading import Lock
from datetime import datetime
import os, io, traceback, math
from Inference import ContextualPDFClassifier
import pypandoc


load_dotenv()

ALLOWED_EXTENSIONS = {"pdf", "doc", "docx"}
TEN_MB = 10 * 1024 * 1024
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "2"))
PORT = int(os.environ.get("PORT", "5050"))  # Custom port (default 5050)

app = Flask(__name__, instance_relative_config=True)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", os.urandom(24)),
    UPLOAD_FOLDER=os.environ.get("UPLOAD_FOLDER", os.path.join(app.instance_path, "uploads")),
)
Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

_jobs = {}
_lock = Lock()
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

_classifier = ContextualPDFClassifier()
RESULTS_FOLDER = os.path.join(app.instance_path, "results")
Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)


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
        # run contextual classifier
        basename = os.path.basename(filepath)
        result_pdf = os.path.join(RESULTS_FOLDER, basename)
        result_json = os.path.splitext(result_pdf)[0] + ".json"

        classification = _classifier.classify_pdf_contextually(
            pdf_path=filepath,
            output_json=result_json,
            output_pdf=result_pdf
        )

        pages = classification["final_classification"]["final_categories"]
        _set_job(
            job_id,
            status="SUCCESS",
            finished_at=datetime.utcnow().isoformat() + "Z",
            result={
                "classified_pdf": os.path.basename(result_pdf),
                "json_result": os.path.basename(result_json),
                "categories": pages
            }
        )
    except Exception:
        _set_job(job_id, status="FAILED", finished_at=datetime.utcnow().isoformat() + "Z", error=traceback.format_exc())



def _paginate(items, page: int, per_page: int):
    total = len(items)
    per_page = max(1, min(per_page, 100))
    total_pages = max(1, math.ceil(total / per_page))
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page
    return items[start:end], {
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
        "has_prev": page > 1,
        "has_next": page < total_pages,
        "prev_page": page - 1 if page > 1 else None,
        "next_page": page + 1 if page < total_pages else None,
    }

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
            flash(f"Only PDF / DOC / DOCX are allowed: '{f.filename}' was rejected")
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
        original_name = secure_filename(f.filename)
        ext = original_name.rsplit(".", 1)[1].lower()

        # Save original file first (UUID prefix to avoid collisions)
        stored_original = f"{job_id}_{original_name}"
        dest_original = os.path.join(app.config["UPLOAD_FOLDER"], stored_original)
        f.stream.seek(0, os.SEEK_SET)
        f.save(dest_original)

        # If DOC/DOCX, convert to PDF via LibreOffice headless
        pdf_path = dest_original
        converted_from = None
        if ext in {"doc", "docx"}:
            converted_from = ext
            pdf_basename = f"{job_id}_{os.path.splitext(original_name)[0]}.pdf"
            try:
                pdf_path = convert_to_pdf(dest_original, app.config["UPLOAD_FOLDER"], desired_name=pdf_basename)
            except Exception as e:
                flash(f"Failed to convert {original_name} to PDF: {e}")
                return redirect(url_for("index"))

        with _lock:
            _jobs[job_id] = {
                "id": job_id,
                "status": "PENDING",
                "created_at": datetime.utcnow().isoformat() + "Z",
                "stored_file": os.path.basename(pdf_path),   # stored PDF filename
                "original_file": original_name,               # the user-uploaded name
                "source_file": os.path.basename(dest_original),
                "converted_from": converted_from,
            }
        _executor.submit(_preprocess_pdf, job_id, pdf_path)
        created_jobs.append({"job_id": job_id, "filename": original_name})

    return render_template("success.html", files=created_jobs)


@app.get("/jobs")
def jobs_page():
    # Server-render the current page immediately so the table isn’t empty on first load
    try:
        page = int(request.args.get("page", 1))
    except ValueError:
        page = 1
    try:
        per_page = int(request.args.get("per_page", 10))
    except ValueError:
        per_page = 10

    jobs_full = _snapshot_jobs()
    jobs_page_items, meta = _paginate(jobs_full, page, per_page)
    return render_template("jobs.html", jobs=jobs_page_items, meta=meta)


@app.get("/api/jobs")
def api_list_jobs():
    # JSON pagination for the dashboard
    try:
        page = int(request.args.get("page", 1))
    except ValueError:
        page = 1
    try:
        per_page = int(request.args.get("per_page", 10))
    except ValueError:
        per_page = 10

    jobs_full = _snapshot_jobs()
    items, meta = _paginate(jobs_full, page, per_page)
    return jsonify({"items": items, "meta": meta})


@app.get("/api/jobs/<job_id>")
def api_get_job(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404
    return jsonify(job)

@app.get("/results/<filename>")
def serve_result_file(filename):
    result_path = os.path.join(RESULTS_FOLDER, filename)
    if not os.path.exists(result_path):
        return "Result not found", 404
    return send_from_directory(RESULTS_FOLDER, filename)

@app.get("/api/jobs/<job_id>/result")
def api_get_result(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job_id"}), 404
    if job.get("status") != "SUCCESS":
        return jsonify({"error": f"Job not complete (status={job.get('status')})"}), 409
    return jsonify(job["result"])

def convert_to_pdf(src_path: str, out_dir: str, desired_name: str | None = None) -> str:
    """
    Convert DOCX → PDF using Pandoc (good for text-heavy docs).
    Layout may differ; needs pandoc + wkhtmltopdf/xelatex.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_name = desired_name or (os.path.splitext(os.path.basename(src_path))[0] + ".pdf")
    out_path = os.path.join(out_dir, out_name)
    pypandoc.convert_file(src_path, "pdf", outputfile=out_path, extra_args=["--standalone"])
    return out_path

if __name__ == "__main__":
    app.run(debug=True, port=PORT)