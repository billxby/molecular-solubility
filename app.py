import json
import os
from pathlib import Path
from flask import Flask, render_template, jsonify

app = Flask(__name__)

DATA_PATH = Path(__file__).resolve().parent / "precomputed.json"
data = None


def _load():
    global data
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found — run 'python3 precompute.py' first"
        )
    data = json.loads(DATA_PATH.read_text())


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/molecules")
def api_molecules():
    return jsonify(data["molecules"])


@app.route("/api/molecule/<int:idx>")
def api_molecule(idx):
    detail = data["details"].get(str(idx))
    if detail is None:
        return jsonify({"error": "index out of range"}), 404
    return jsonify(detail)


if __name__ == "__main__":
    print("Loading precomputed data ...")
    _load()
    print(f"Ready — {len(data['molecules'])} molecules.")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 4000)))
