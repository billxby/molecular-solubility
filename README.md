Molecular Solubility Prediction

Comparing three neural network architectures for predicting aqueous solubility (log S) on the ESOL dataset: a standard MLP on Morgan fingerprints, a Graph Convolutional Network (GCN), and a Graph Attention Network (GAT).

Project structure:

training.ipynb
  Main notebook. Trains all three models, evaluates them on the test set (RMSE, MAE, R²), and plots loss curves and predicted vs actual scatter plots. After training, run the **save trained models** cell to write weights and histories under `models/` (`mlp.pt`, `gcn.pt`, `gat.pt`, `training_histories.json`). To avoid retraining on a later run, set `LOAD_SAVED_WEIGHTS = True` in the optional load cell (right after the data cell) and skip the three training sections; then run evaluation and plotting as usual.

models/
  Trained checkpoints produced by the notebook (contents ignored by git; the folder is kept with `.gitkeep`).

extra-insights.ipynb
  Trains a GAT and visualizes attention weights on a few example molecules to see which atoms the model focuses on.

dataset/
  dataset_util.py — loads the ESOL dataset, handles train/val/test splits, generates Morgan fingerprints, and defines evaluation metrics (RMSE, MAE, R²).
  dataset-exploration.ipynb — explores the dataset: target distribution, molecule sizes, sample SMILES.

nn/
  nn_util.py — MLP model definition and training helpers.
  implement-nn.ipynb — builds a simple neural network from scratch (no nn.Module) to show how forward pass, backprop, and gradient descent work.

gnns/
  gnn_util.py — GCN and GAT model definitions and training helpers.

data/
  ESOL dataset files (loaded automatically by PyTorch Geometric).
