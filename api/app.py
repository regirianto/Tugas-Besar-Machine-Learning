"""
API + Web Server - Tugas Besar ML Prediksi Curah Hujan
Memuat model hasil Google Colab dari ``model/model_cuaca_kemayoran.pth``
(atau fallback ``model/model.pkl``) dan melayani web app di ``/``.

Jalankan:
    pip install -r requirements.txt
    python app.py
    buka http://localhost:5000
"""
from __future__ import annotations

import json
import os
import pickle
import time
import warnings
import zipfile
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "model"
BUNDLE_PATH = MODEL_DIR / "model_cuaca_kemayoran.pth"
LEGACY_MODEL_PATH = MODEL_DIR / "model.pkl"
FEATURES_PATH = MODEL_DIR / "features.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

DEFAULT_THRESHOLD = 0.5

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
# Hindari bentrok sintaks Vue `{{ }}` dengan Jinja
app.jinja_env.variable_start_string = "(("
app.jinja_env.variable_end_string = "))"

# Cache sederhana: reload hanya kalau file artefak berubah.
_CACHE: dict[str, object] = {}

# Model dilatih di Colab (scikit-learn 1.6.x) lalu dibuka di mesin dengan
# scikit-learn lebih baru. SimpleImputer 1.8 mengganti atribut internal
# ``_fill_dtype`` menjadi ``_fit_dtype``; patch kecil ini menjaga model lama
# tetap bisa memprediksi.
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:  # pragma: no cover
    pass


def _patch_model_compat(model) -> None:
    steps = getattr(model, "steps", None)
    if not steps:
        return
    for _, estimator in steps:
        if type(estimator).__name__ == "SimpleImputer" and not hasattr(estimator, "_fill_dtype"):
            try:
                estimator._fill_dtype = getattr(estimator, "_fit_dtype", float)
            except Exception:  # pragma: no cover
                pass


def _artifact_candidates() -> list[Path]:
    return [p for p in (BUNDLE_PATH, LEGACY_MODEL_PATH) if p.exists()]


def _read_artifact(path: Path):
    """Baca bundle ``.pth`` (pickle) atau artefak joblib lama."""
    with open(path, "rb") as f:
        head = f.read(4)

    # torch.save(...) menghasilkan arsip ZIP. Dukungan ini opsional agar file
    # .pth yang benar-benar dibuat dengan torch tetap bisa dibuka.
    if head[:2] == b"PK":
        try:
            import torch  # type: ignore
            return torch.load(path, map_location="cpu", weights_only=False)
        except ImportError:
            # Bundle tanpa tensor tetap bisa dibaca dari arsip torch.save
            # tanpa menginstal torch: ambil pickle stream-nya.
            try:
                with zipfile.ZipFile(path) as zf:
                    names = [n for n in zf.namelist()
                             if n == "data.pkl" or n.endswith("/data.pkl")]
                    if not names:
                        raise KeyError("data.pkl tidak ditemukan")
                    with zf.open(names[0]) as f:
                        return pickle.load(f)
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "File .pth adalah arsip PyTorch yang berisi tensor. "
                    "Jalankan `pip install torch` atau gunakan bundle pickle "
                    "dari notebooks/02-part3-export-pth.ipynb."
                ) from exc

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return joblib.load(path)


def _normalize_bundle(obj, path: Path) -> dict:
    """Ubah objek lama/bundle baru menjadi struktur yang seragam."""
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        features = obj.get("features")
        metadata = obj.get("metadata") or {}
        threshold = obj.get("threshold", metadata.get("threshold", DEFAULT_THRESHOLD))
        created = obj.get("created_utc", "")
    else:
        model = obj
        features = None
        metadata = {}
        threshold = DEFAULT_THRESHOLD
        created = ""

    _patch_model_compat(model)

    if features is None:
        if FEATURES_PATH.exists():
            features = joblib.load(FEATURES_PATH)
        else:
            features = list(getattr(model, "feature_names_in_", []))

    if not metadata and METADATA_PATH.exists():
        with open(METADATA_PATH, encoding="utf-8") as f:
            metadata = json.load(f)

    if not created:
        created = time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(path.stat().st_mtime))

    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = DEFAULT_THRESHOLD

    return {
        "model": model,
        "features": list(features),
        "threshold": threshold,
        "metadata": metadata,
        "artifact": path.name,
        "model_name": metadata.get("model", "model"),
        "created_utc": created,
    }


def load_bundle() -> dict | None:
    """Muat artefak terbaru; return ``None`` bila belum ada."""
    candidates = _artifact_candidates()
    if not candidates:
        return None

    path = candidates[0]
    stat = path.stat()
    cache_key = f"{path}:{stat.st_mtime_ns}:{stat.st_size}"
    if _CACHE.get("key") == cache_key:
        return _CACHE["bundle"]  # type: ignore[return-value]

    bundle = _normalize_bundle(_read_artifact(path), path)
    _CACHE.clear()
    _CACHE["key"] = cache_key
    _CACHE["bundle"] = bundle
    return bundle


def _ready_state() -> tuple[dict | None, str]:
    bundle = load_bundle()
    if bundle is None:
        return None, "model .pth/.pkl belum ada - jalankan export dari Colab"
    return bundle, f"{bundle['artifact']}: siap"


@app.errorhandler(500)
def internal_error(error):  # pragma: no cover
    return jsonify({"error": "terjadi kesalahan server", "detail": str(error)}), 500


@app.route("/")
def index():
    bundle, status = _ready_state()
    return render_template(
        "index.html",
        status=status,
        ready=bundle is not None,
        threshold=bundle["threshold"] if bundle else DEFAULT_THRESHOLD,
        artifact=bundle["artifact"] if bundle else "-",
        model_name=bundle["model_name"] if bundle else "-",
    )


@app.route("/health")
def health():
    bundle, status = _ready_state()
    return jsonify({
        "status": "ok" if bundle else "model_missing",
        "message": status,
        "artifact": bundle["artifact"] if bundle else None,
        "model": bundle["model_name"] if bundle else None,
        "threshold": bundle["threshold"] if bundle else None,
        "features": bundle["features"] if bundle else [],
    }), (200 if bundle else 503)


@app.route("/metadata")
def metadata():
    bundle, status = _ready_state()
    if bundle is None:
        return jsonify({"error": status}), 503
    return jsonify({
        "artifact": bundle["artifact"],
        "created_utc": bundle["created_utc"],
        "threshold": bundle["threshold"],
        "features": bundle["features"],
        "metadata": bundle["metadata"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    bundle, status = _ready_state()
    if bundle is None:
        return jsonify({
            "error": "Model belum ada. Jalankan notebooks/02-part3-export-pth.ipynb "
                     "di Colab, download model_cuaca_kemayoran.pth, lalu taruh di folder model/."
        }), 503

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "body kosong / bukan JSON."}), 400

    features = bundle["features"]
    missing = [f for f in features if payload.get(f) is None]
    if missing:
        return jsonify({"error": f"input kurang: {', '.join(missing)}"}), 400

    try:
        values = {f: float(payload[f]) for f in features}
    except (TypeError, ValueError):
        return jsonify({"error": "semua nilai fitur harus berupa angka"}), 400

    if "month" in values:
        if not 1 <= values["month"] <= 12:
            return jsonify({"error": "month harus di antara 1 dan 12"}), 400
        values["month"] = int(values["month"])

    sample = pd.DataFrame([values])
    try:
        proba = float(bundle["model"].predict_proba(sample)[0][1])
    except Exception as exc:  # model tidak punya predict_proba / mismatch fitur
        return jsonify({"error": f"gagal memprediksi: {exc}"}), 500

    threshold = bundle["threshold"]
    prediction = "HUJAN LEBAT" if proba >= threshold else "TIDAK HUJAN LEBAT"

    return jsonify({
        "prediction": prediction,
        "prob": round(proba, 3),
        "threshold": threshold,
        "model": bundle["model_name"],
        "artifact": bundle["artifact"],
        "input": values,
        "note": f"positif = RR >= 20 mm hujan lebat (proba >= {threshold:.2f})",
    })


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
