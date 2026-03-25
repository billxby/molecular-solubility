import json, io, base64, math
from pathlib import Path

import torch
import numpy as np
from flask import Flask, render_template, jsonify

from rdkit import Chem
from rdkit.Chem import Draw, rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.datasets import MoleculeNet

# project imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gnns.gnn_util import GCNModel, GATModel
from nn.nn_util import MLPModel
from dataset.dataset_util import smiles_to_fingerprint, split_dataset


# globals

app = Flask(__name__)

MODEL_DIR = Path(__file__).resolve().parent / "models"
DEVICE = "cpu"

dataset = None          # full MoleculeNet ESOL
train_set = None
test_set = None
mlp_model = None
gcn_model = None
gat_model = None
molecules = []          # list of dicts for the table


# startup – load data & models


def _mol_to_svg(smiles: str, size=(250, 180)) -> str:
    """Return an SVG string of the 2-D depiction."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _load_models():
    global dataset, train_set, test_set, mlp_model, gcn_model, gat_model, molecules

    # dataset
    dataset = MoleculeNet(root="data/", name="ESOL")
    train_set, test_set = split_dataset(dataset, test_ratio=0.2, seed=42)
    in_channels = dataset.num_node_features

    # models
    mlp_model = MLPModel(in_features=1024, hidden=128)
    gcn_model = GCNModel(in_channels=in_channels, hidden=128)
    gat_model = GATModel(in_channels=in_channels, hidden=128, heads=4)

    mlp_path = MODEL_DIR / "mlp.pt"
    gcn_path = MODEL_DIR / "gcn.pt"
    gat_path = MODEL_DIR / "gat.pt"

    models_loaded = True
    for p in [mlp_path, gcn_path, gat_path]:
        if not p.exists():
            print(f"WARNING: {p} not found – predictions will be random (untrained weights)")
            models_loaded = False
            break

    if models_loaded:
        mlp_model.load_state_dict(torch.load(mlp_path, map_location=DEVICE))
        gcn_model.load_state_dict(torch.load(gcn_path, map_location=DEVICE))
        gat_model.load_state_dict(torch.load(gat_path, map_location=DEVICE))

    mlp_model.eval()
    gcn_model.eval()
    gat_model.eval()

    # build molecule list (use test set so predictions are on unseen data)
    for i, data in enumerate(test_set):
        smiles = data.smiles
        actual = data.y.item()
        molecules.append({
            "id": i,
            "smiles": smiles,
            "actual": round(actual, 3),
            "name": smiles[:40],
        })



# inference helpers


@torch.no_grad()
def _predict_single(idx: int):
    """Return dict with predictions from all three models for test_set[idx]."""
    data = test_set[idx]
    smiles = data.smiles

    # MLP – fingerprint input
    fp = torch.tensor(smiles_to_fingerprint(smiles, n_bits=1024, radius=2)).unsqueeze(0)
    mlp_pred = mlp_model(fp).item()

    # GCN
    gcn_pred = gcn_model(data.x, data.edge_index, torch.zeros(data.x.size(0), dtype=torch.long)).item()

    # GAT
    gat_pred = gat_model(data.x, data.edge_index, torch.zeros(data.x.size(0), dtype=torch.long)).item()

    return {
        "mlp": round(mlp_pred, 4),
        "gcn": round(gcn_pred, 4),
        "gat": round(gat_pred, 4),
        "actual": round(data.y.item(), 4),
    }


@torch.no_grad()
def _get_attention_svg(idx: int, size=(500, 400)):
    """
    Compute GAT attention weights for molecule at test_set[idx] and render
    an SVG with atoms coloured by aggregated attention.
    """
    data = test_set[idx]
    smiles = data.smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""

    # get attention from both layers
    (ei1, att1), (ei2, att2) = gat_model.get_attention(data.x.float(), data.edge_index)

    # aggregate attention: for each atom, sum incoming attention weights
    num_atoms = data.x.size(0)

    # use layer-2 attention (closer to output, more task-relevant)
    # att2 shape: (num_edges, heads) – average over heads then aggregate per target node
    att_mean = att2.mean(dim=1)  # (num_edges,)
    atom_att = torch.zeros(num_atoms)
    target_nodes = ei2[1]
    atom_att.scatter_add_(0, target_nodes, att_mean)

    # normalize to [0, 1]
    att_min = atom_att.min()
    att_max = atom_att.max()
    if att_max - att_min > 1e-8:
        atom_att = (atom_att - att_min) / (att_max - att_min)
    else:
        atom_att = torch.zeros(num_atoms)

    att_np = atom_att.numpy()

    # map to colours: low attention = light blue, high attention = red
    atom_colors = {}
    bond_colors = {}
    atom_radii = {}
    highlight_atoms = list(range(num_atoms))
    for a_idx in range(num_atoms):
        v = float(att_np[a_idx])
        # blue (0) -> white (0.5) -> red (1)
        if v < 0.5:
            t = v * 2
            r, g, b = 0.7 * (1 - t) + t, 0.85 * (1 - t) + t, 1.0
        else:
            t = (v - 0.5) * 2
            r, g, b = 1.0, 1.0 * (1 - t), 0.7 * (1 - t)
        atom_colors[a_idx] = (r, g, b)
        atom_radii[a_idx] = 0.3 + 0.3 * v

    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    opts = drawer.drawOptions()
    opts.fillHighlights = True
    opts.continuousHighlight = False

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightBonds=[],
        highlightBondColors=bond_colors,
        highlightAtomRadii=atom_radii,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    # also return per-atom attention values for the legend
    atom_info = []
    for a_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(a_idx)
        atom_info.append({
            "idx": a_idx,
            "symbol": atom.GetSymbol(),
            "attention": round(float(att_np[a_idx]), 4),
        })

    return svg, atom_info



# routes


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/molecules")
def api_molecules():
    return jsonify(molecules)


@app.route("/api/molecule/<int:idx>")
def api_molecule(idx):
    if idx < 0 or idx >= len(molecules):
        return jsonify({"error": "index out of range"}), 404

    data = test_set[idx]
    smiles = data.smiles

    preds = _predict_single(idx)
    structure_svg = _mol_to_svg(smiles, size=(300, 220))
    attention_svg, atom_info = _get_attention_svg(idx, size=(500, 400))

    return jsonify({
        "smiles": smiles,
        "actual": preds["actual"],
        "predictions": {
            "MLP (Fingerprint)": preds["mlp"],
            "GCN": preds["gcn"],
            "GAT": preds["gat"],
        },
        "structure_svg": structure_svg,
        "attention_svg": attention_svg,
        "atom_attention": atom_info,
    })



# main

if __name__ == "__main__":
    print("Loading dataset and models …")
    _load_models()
    print(f"Ready – {len(molecules)} test molecules loaded.")
    app.run(debug=True, port=5050)
