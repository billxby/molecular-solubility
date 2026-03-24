import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool


# ---- GCN ----

class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden=128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = global_mean_pool(x, batch)
        return self.out(x)


# ---- GAT ----

class GATModel(nn.Module):
    def __init__(self, in_channels, hidden=128, heads=4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden // heads, heads=heads)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = GATConv(hidden, hidden // heads, heads=heads)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, batch):
        x = x.float()
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.relu(self.bn2(self.conv2(x, edge_index)))
        x = global_mean_pool(x, batch)
        return self.out(x)

    def get_attention(self, x, edge_index):
        """Run GAT conv layers and return attention weights from both layers."""
        x = x.float()
        x, (ei1, att1) = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.relu(self.bn1(x))
        x, (ei2, att2) = self.conv2(x, edge_index, return_attention_weights=True)
        return (ei1, att1), (ei2, att2)


# ---- training helpers ----

def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    n = 0
    for batch in loader:
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
        loss = loss_fn(pred, batch.y.squeeze(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / n


@torch.no_grad()
def eval_epoch(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    n = 0
    for batch in loader:
        pred = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
        loss = loss_fn(pred, batch.y.squeeze(-1))
        total_loss += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total_loss / n


def run_training(model, train_loader, val_loader=None, epochs=200, lr=0.001, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {'train_loss': []}
    if val_loader is not None:
        history['val_loss'] = []
    for epoch in range(1, epochs + 1):
        t_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        history['train_loss'].append(t_loss)
        msg = f'  epoch {epoch:>3d}  train {t_loss:.4f}'
        if val_loader is not None:
            v_loss = eval_epoch(model, val_loader, loss_fn)
            history['val_loss'].append(v_loss)
            msg += f'  val {v_loss:.4f}'

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(msg)

    return history


# ---- inference helper ----

@torch.no_grad()
def get_predictions(model, loader):
    model.eval()
    preds, targets = [], []
    for batch in loader:
        pred = model(batch.x, batch.edge_index, batch.batch).squeeze(-1)
        preds.append(pred)
        targets.append(batch.y.squeeze(-1))
    return torch.cat(preds), torch.cat(targets)
