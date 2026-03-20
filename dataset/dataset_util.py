import torch
import numpy as np
from functools import lru_cache
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


@lru_cache(maxsize=32)
def _morgan_generator(radius: int, n_bits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


# ---- loading ----

def load_esol(root='data/'):
    dataset = MoleculeNet(root=root, name='ESOL')
    return dataset


# ---- splits ----

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    torch.manual_seed(seed)
    n = len(dataset)
    perm = torch.randperm(n)

    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]

    return dataset[train_idx], dataset[val_idx], dataset[test_idx]


def make_loaders(train_set, val_set, test_set, batch_size=64):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, val_loader, test_loader


# ---- fingerprints (for MLP baseline) ----

def smiles_to_fingerprint(smiles, n_bits=1024, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    fpgen = _morgan_generator(radius, n_bits)
    return np.asarray(fpgen.GetFingerprintAsNumPy(mol), dtype=np.float32)

def get_fingerprints(dataset, n_bits=1024, radius=2):
    # each data object has a .smiles attribute from MoleculeNet
    fps = []
    targets = []
    for data in dataset:
        smiles = data.smiles
        fp = smiles_to_fingerprint(smiles, n_bits, radius)
        fps.append(fp)
        targets.append(data.y.item())

    X = torch.tensor(np.array(fps), dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    return X, y


# ---- metrics ----

def rmse(preds, targets):
    return torch.sqrt(torch.mean((preds - targets) ** 2)).item()

def mae(preds, targets):
    return torch.mean(torch.abs(preds - targets)).item()

def r_squared(preds, targets):
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)
    return (1 - ss_res / ss_tot).item()
