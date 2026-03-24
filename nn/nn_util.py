import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- MLP for fingerprint-based regression ----

class MLPModel(nn.Module):
    def __init__(self, in_features=1024, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.out(x)


# ---- training helpers ----

def train_epoch_mlp(model, X, y, optimizer, loss_fn, batch_size=64):
    model.train()
    n = X.size(0)
    perm = torch.randperm(n)
    total_loss = 0
    count = 0
    for i in range(0, n, batch_size):
        idx = perm[i:i+batch_size]
        xb, yb = X[idx], y[idx]
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)
        count += len(idx)
    return total_loss / count


@torch.no_grad()
def eval_mlp(model, X, y, loss_fn):
    model.eval()
    pred = model(X)
    return loss_fn(pred, y).item()


def run_training_mlp(model, X_train, y_train, X_val=None, y_val=None, epochs=200, lr=0.001, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {'train_loss': []}
    if X_val is not None:
        history['val_loss'] = []
    for epoch in range(1, epochs + 1):
        t_loss = train_epoch_mlp(model, X_train, y_train, optimizer, loss_fn)
        history['train_loss'].append(t_loss)
        msg = f'  epoch {epoch:>3d}  train {t_loss:.4f}'
        if X_val is not None:
            v_loss = eval_mlp(model, X_val, y_val, loss_fn)
            history['val_loss'].append(v_loss)
            msg += f'  val {v_loss:.4f}'

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(msg)

    return history


@torch.no_grad()
def get_predictions_mlp(model, X):
    model.eval()
    return model(X)
