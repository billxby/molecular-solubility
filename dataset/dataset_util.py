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

def split_dataset(dataset, test_ratio=0.2, seed=42):
    torch.manual_seed(seed)
    n = len(dataset)
    perm = torch.randperm(n)

    test_size = int(n * test_ratio)
    train_idx = perm[:n - test_size]
    test_idx = perm[n - test_size:]

    return dataset[train_idx], dataset[test_idx]


def kfold_indices(n, k=5, seed=42):
    """Return list of (train_indices, val_indices) for k-fold CV."""
    torch.manual_seed(seed)
    perm = torch.randperm(n)
    fold_size = n // k
    folds = []
    for i in range(k):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < k - 1 else n
        val_idx = perm[val_start:val_end]
        train_idx = torch.cat([perm[:val_start], perm[val_end:]])
        folds.append((train_idx, val_idx))
    return folds


def make_loaders(train_set, test_set, batch_size=64):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader


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
