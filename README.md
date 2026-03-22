Molecular Solubility Prediction

Comparing three neural network architectures for predicting aqueous solubility (log S) on the ESOL dataset: a standard MLP on Morgan fingerprints, a Graph Convolutional Network (GCN), and a Graph Attention Network (GAT).

Project structure:

training.ipynb
  Main notebook. Trains all three models, evaluates them on the test set (RMSE, MAE, R²), and plots loss curves and predicted vs actual scatter plots.

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
