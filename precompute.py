"""
Precompute all predictions, structure SVGs, and attention maps for the test set.

Run locally (requires torch, rdkit, torch_geometric):
    python3 precompute.py

Produces: precomputed.json (~2-4 MB)
"""

import json, sys
from pathlib import Path

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.datasets import MoleculeNet

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gnns.gnn_util import GCNModel, GATModel
from nn.nn_util import MLPModel
from dataset.dataset_util import smiles_to_fingerprint, split_dataset

MODEL_DIR = Path(__file__).resolve().parent / "models"
OUT_PATH = Path(__file__).resolve().parent / "precomputed.json"


def mol_to_svg(smiles, size=(300, 220)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


@torch.no_grad()
def get_attention_svg(gat_model, data, size=(500, 400)):
    smiles = data.smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", []

    (ei1, att1), (ei2, att2) = gat_model.get_attention(data.x.float(), data.edge_index)
    num_atoms = data.x.size(0)
    att_mean = att2.mean(dim=1)
    atom_att = torch.zeros(num_atoms)
    atom_att.scatter_add_(0, ei2[1], att_mean)

    att_min, att_max = atom_att.min(), atom_att.max()
    if att_max - att_min > 1e-8:
        atom_att = (atom_att - att_min) / (att_max - att_min)
    else:
        atom_att = torch.zeros(num_atoms)

    att_np = atom_att.numpy()

    atom_colors, atom_radii = {}, {}
    for a_idx in range(num_atoms):
        v = float(att_np[a_idx])
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
        highlightAtoms=list(range(num_atoms)),
        highlightAtomColors=atom_colors,
        highlightBonds=[],
        highlightBondColors={},
        highlightAtomRadii=atom_radii,
    )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    atom_info = []
    for a_idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(a_idx)
        atom_info.append({
            "idx": a_idx,
            "symbol": atom.GetSymbol(),
            "attention": round(float(att_np[a_idx]), 4),
        })

    return svg, atom_info


def main():
    print("Loading dataset ...")
    dataset = MoleculeNet(root="data/", name="ESOL")
    _, test_set = split_dataset(dataset, test_ratio=0.2, seed=42)
    in_channels = dataset.num_node_features

    print("Loading models ...")
    mlp = MLPModel(in_features=1024, hidden=128)
    gcn = GCNModel(in_channels=in_channels, hidden=128)
    gat = GATModel(in_channels=in_channels, hidden=128, heads=4)

    models_loaded = True
    for p in [MODEL_DIR / "mlp.pt", MODEL_DIR / "gcn.pt", MODEL_DIR / "gat.pt"]:
        if not p.exists():
            print(f"WARNING: {p} not found – using untrained weights")
            models_loaded = False
            break

    if models_loaded:
        mlp.load_state_dict(torch.load(MODEL_DIR / "mlp.pt", map_location="cpu"))
        gcn.load_state_dict(torch.load(MODEL_DIR / "gcn.pt", map_location="cpu"))
        gat.load_state_dict(torch.load(MODEL_DIR / "gat.pt", map_location="cpu"))

    mlp.eval()
    gcn.eval()
    gat.eval()

    molecules = []  # for the list endpoint
    details = {}    # keyed by id, for the detail endpoint

    print(f"Precomputing {len(test_set)} molecules ...")
    for i, data in enumerate(test_set):
        smiles = data.smiles
        actual = round(data.y.item(), 4)

        # predictions
        fp = torch.tensor(smiles_to_fingerprint(smiles)).unsqueeze(0)
        mlp_pred = round(mlp(fp).item(), 4)
        batch = torch.zeros(data.x.size(0), dtype=torch.long)
        gcn_pred = round(gcn(data.x, data.edge_index, batch).item(), 4)
        gat_pred = round(gat(data.x, data.edge_index, batch).item(), 4)

        # SVGs
        structure_svg = mol_to_svg(smiles)
        attention_svg, atom_info = get_attention_svg(gat, data)

        molecules.append({
            "id": i,
            "smiles": smiles,
            "actual": round(actual, 3),
        })

        details[str(i)] = {
            "smiles": smiles,
            "actual": actual,
            "predictions": {
                "MLP (Fingerprint)": mlp_pred,
                "GCN": gcn_pred,
                "GAT": gat_pred,
            },
            "structure_svg": structure_svg,
            "attention_svg": attention_svg,
            "atom_attention": atom_info,
        }

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(test_set)}")

    output = {"molecules": molecules, "details": details}
    OUT_PATH.write_text(json.dumps(output))
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"Wrote {OUT_PATH} ({size_mb:.1f} MB, {len(molecules)} molecules)")


if __name__ == "__main__":
    main()
