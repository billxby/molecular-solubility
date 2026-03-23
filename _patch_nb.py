import json

path = "training.ipynb"
with open(path, encoding="utf-8") as f:
    nb = json.load(f)

new_src = """# Optional: load weights from models/ so you can skip the three training sections below.
LOAD_SAVED_WEIGHTS = False

if LOAD_SAVED_WEIGHTS:
    import json
    from pathlib import Path

    MODEL_DIR = Path('models')
    required = [MODEL_DIR / 'mlp.pt', MODEL_DIR / 'gcn.pt', MODEL_DIR / 'gat.pt', MODEL_DIR / 'training_histories.json']
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError('missing ' + ', '.join(str(p) for p in missing) + ' — train and run the save cell first.')

    mlp = MLPModel(in_features=1024, hidden=128)
    mlp.load_state_dict(torch.load(MODEL_DIR / 'mlp.pt', map_location='cpu'))

    gcn = GCNModel(in_channels=dataset.num_node_features, hidden=128)
    gcn.load_state_dict(torch.load(MODEL_DIR / 'gcn.pt', map_location='cpu'))

    gat = GATModel(in_channels=dataset.num_node_features, hidden=128, heads=4)
    gat.load_state_dict(torch.load(MODEL_DIR / 'gat.pt', map_location='cpu'))

    with open(MODEL_DIR / 'training_histories.json') as f:
        histories = json.load(f)
    mlp_history = histories['mlp']
    gcn_history = histories['gcn']
    gat_history = histories['gat']
"""

for c in nb["cells"]:
    if c["cell_type"] == "code" and any(
        "LOAD_SAVED_WEIGHTS" in line for line in c.get("source", [])
    ):
        c["source"] = [ln + "\n" for ln in new_src.splitlines()]
        break

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("patched load cell")
