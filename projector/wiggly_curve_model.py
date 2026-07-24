import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# Synthetic dataset: label = above/below wiggly curve
# ============================================================

class WigglyCurveDataset(Dataset):
    """
    Synthetic 2D dataset.
    Label = 1 if point is above the boundary(x1), else 0.
    """

    def __init__(self, n=6000, noise=0.05, seed=0, boundary_fn=None):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        if boundary_fn is None:
            raise ValueError("boundary_fn must be provided")

        self.boundary_fn = boundary_fn

        x1 = torch.rand(n, generator=g) * 2 * math.pi
        x2 = torch.rand(n, generator=g) * 2 - 1.0

        f = self.boundary(x1)
        y = (x2 > f).long()

        X = torch.stack([x1, x2], dim=1)
        noise_term = torch.randn(X.size(), generator=g)
        X += noise * noise_term

        self.X = X.float()
        self.y = y.long()

    def boundary(self, x1):
        return self.boundary_fn(x1)

    def __len__(self):
        return self.y.numel()

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# Model functions ====================================== Begin

class Restrict(nn.Module):  # Q*
    def __init__(self, D, Dc):
        super().__init__()
        self.proj = nn.Linear(D, Dc, bias=False)

    def forward(self, x):
        return self.proj(x)


class Prolong(nn.Module):  # Q
    def __init__(self, Dc, D):
        super().__init__()
        self.proj = nn.Linear(Dc, D, bias=False)

    def forward(self, z):
        return self.proj(z)


class Projector(nn.Module):
    def __init__(self, D, Dc, eps=1e-6):
        super().__init__()
        self.Qs = Restrict(D, Dc)
        self.Q  = Prolong(Dc, D)
        self.eps = eps

    def forward(self, x):
        """
        x: [B, T, D]
        """
        z = self.Qs(x)  # [B, T, Dc]

        Wq  = self.Q.proj.weight      # [D, Dc]
        Wqs = self.Qs.proj.weight     # [Dc, D]
        A = Wqs @ Wq                  # [Dc, Dc]
        A = A + self.eps * torch.eye(A.size(0), device=A.device)

        B, T, Dc = z.shape
        z_flat = z.reshape(-1, Dc)          # [(B*T), Dc]

        u = torch.linalg.solve(A, z_flat.T) # [Dc, (B*T)]
        u = u.T.reshape(B, T, Dc)

        return self.Q(u)  # [B, T, D]

class ToyNet(nn.Module):
    def __init__(self, input_dim=2, hidden=256, num_hidden_layers=2, Dc=16, use_projector=False, n_proj_iters=2):
        super().__init__()

        self.use_projector = use_projector
        self.n_proj_iters = n_proj_iters

        # Hidden layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden))
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden, hidden))
        self.hidden_layers = nn.ModuleList(layers)

        # Output layer
        self.out = nn.Linear(hidden, 2)

        # Projector (shared)
        if use_projector:
            self.proj = Projector(hidden, Dc)
            self.logit_alpha_h = nn.Parameter(torch.tensor(0.0))
        else:
            self.proj = None

    def apply_proj(self, h):
        if self.proj is None or self.n_proj_iters <= 0:
            return h

        alpha = torch.sigmoid(self.logit_alpha_h)

        for _ in range(self.n_proj_iters):
            h_proj = self.proj(h.unsqueeze(1)).squeeze(1)
            h = alpha * h + (1.0 - alpha) * h_proj

        return h

    def forward(self, x):
        h = x
        states = []
        for layer in self.hidden_layers:
            h = torch.tanh(layer(h))
            h = self.apply_proj(h)
            states.append(h)

        # Stack layer outputs as tokens
        H = torch.stack(states, dim=1)   # [B, L, hidden]

        B = x.shape[0]
        L = len(self.hidden_layers)
        D = self.hidden_layers[0].out_features  # more explicit than H.shape[-1]

        assert H.shape == (B, L, D), f"H has shape {H.shape}, expected ({B}, {L}, {D})"
        assert H.ndim == 3

        h = H[:, -1]  # use only the final layer / "reasoning step" (B, D)
        # h = H.mean(dim=1) # averages all layers / reasoning steps.(B, D)
        return self.out(h)

# Model functions ====================================== End
@torch.no_grad()
def eval_loss(model, loader, device="cpu"):
    model.eval()
    total_loss = 0.0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x), y)
        total_loss += loss.item() * y.size(0)
        total += y.size(0)
    return total_loss / total

@torch.no_grad()
def eval_metrics(model, loader, device="cpu"):
    model.eval()
    devs = [torch.cuda.current_device()] if str(device).startswith("cuda") else []
    with torch.random.fork_rng(devices=devs):
        torch.manual_seed(0)

        tp = fp = fn = tn = 0
        total_loss = 0.0
        total = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * y.size(0)
            total += y.size(0)

            tp += ((preds == 1) & (y == 1)).sum().item()
            tn += ((preds == 0) & (y == 0)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)

    return {
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "val_loss": total_loss / total,
    }

@torch.no_grad()
def accuracy(model, loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)

def train(model, train_loader, val_loader,
               epochs=200, lr=1e-3, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    hist = {
        "acc": [],
        "prec": [],
        "rec": [],
        "f1": [],
        "train_loss": [],
        "val_loss": [],
    }

    for _ in range(epochs):
        # ---- train ----
        model.train()
        total_loss = 0.0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * y.size(0)
            total += y.size(0)

        hist["train_loss"].append(total_loss / total)

        # ---- validation metrics ----
        m = eval_metrics(model, val_loader, device)
        for k in ["acc", "prec", "rec", "f1", "val_loss"]:
            hist[k].append(m[k])

    return hist

# === Plot Functions ================================================== Begin

@torch.no_grad()
def _plot_decision_boundary_ax(
        ax,
        model,
        dataset,
        title="",
        device="cpu",
        grid_res=300
):
    model.eval()
    model.to(device)

    X = dataset.X.cpu().numpy()
    y = dataset.y.cpu().numpy()

    x1_min, x1_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
    x2_min, x2_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2

    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, grid_res),
        np.linspace(x2_min, x2_max, grid_res)
    )

    grid = torch.tensor(
        np.c_[xx1.ravel(), xx2.ravel()],
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        logits = model(grid)
        probs = logits[:, 1] - logits[:, 0]

    Z = probs.reshape(xx1.shape).cpu().numpy()

    # True boundary
    x1_curve = np.linspace(x1_min, x1_max, 1000)
    x1_curve_t = torch.tensor(x1_curve, dtype=torch.float32, device=device)
    f_curve = dataset.boundary(x1_curve_t).cpu().numpy()

    ax.plot(
        x1_curve,
        f_curve,
        "--",
        color="brown",
        linewidth=2,
        label="True boundary"
    )

    # Model boundary
    ax.contour(xx1, xx2, Z, levels=[0.0], colors="blue", linewidths=2)

    # Background + points
    ax.contourf(xx1, xx2, (Z > 0), levels=1, alpha=0.35)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolors="k")

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()


def plot_decision_boundaries_side_by_side(
        model_left,
        model_right,
        dataset,
        left_title="No projector",
        right_title="With projector",
        device="cpu",
        grid_res=300
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    _plot_decision_boundary_ax(
        axes[0],
        model_left,
        dataset,
        title=left_title,
        device=device,
        grid_res=grid_res
    )

    _plot_decision_boundary_ax(
        axes[1],
        model_right,
        dataset,
        title=right_title,
        device=device,
        grid_res=grid_res
    )

    plt.tight_layout()
    plt.show()

def plot_metrics_curves(
        hist_plain,
        hist_proj,
        title_fs=16,
        label_fs=13,
        tick_fs=11,
):
    epochs = range(1, len(hist_plain["train_loss"]) + 1)

    fig, axes = plt.subplots(3, 2, figsize=(11, 12))
    axes = axes.ravel()

    plots = [
        ("Val Accuracy", "acc"),
        ("Val Precision", "prec"),
        ("Val Recall", "rec"),
        ("Val F1-score", "f1"),
        ("Train loss", "train_loss"),
        ("Validation loss", "val_loss"),
    ]

    for ax, (title, key) in zip(axes, plots):
        ax.plot(epochs, hist_plain[key], label="Plain", linewidth=2)
        ax.plot(epochs, hist_proj[key], label="Proj", linestyle="--", linewidth=2)

        ax.set_title(title, fontsize=title_fs)
        ax.set_xlabel("Epoch", fontsize=label_fs, labelpad=6)
        ax.set_ylabel(title, fontsize=label_fs, labelpad=6)

        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(True)
        ax.legend(fontsize=label_fs - 1)

    fig.tight_layout()
    plt.show()


def print_metrics(hist_plain, hist_proj):
    df = pd.DataFrame(
        {
            "PLAIN": {
                "accuracy":        hist_plain["acc"][-1],
                "precision":       hist_plain["prec"][-1],
                "recall":          hist_plain["rec"][-1],
                "f1_score":        hist_plain["f1"][-1],
                '':                '',
                "train_loss":      hist_plain["train_loss"][-1],
                "val_loss":        hist_plain["val_loss"][-1],
            },
            "PROJ": {
                "accuracy":        hist_proj["acc"][-1],
                "precision":       hist_proj["prec"][-1],
                "recall":          hist_proj["rec"][-1],
                "f1_score":        hist_proj["f1"][-1],
                '':                '',
                "train_loss":      hist_proj["train_loss"][-1],
                "val_loss":        hist_proj["val_loss"][-1],
            },
        }
    )
    print()
    df = df.apply(pd.to_numeric).round(4).fillna("")
    print(df.to_string())
    print()
# === Plot Functions ================================================== End

def checksum(model):
    s = 0.0
    for p in model.parameters():
        s += p.detach().float().sum().item()
    return s

class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__

# Each sample is one 2D point (X1, X2) with a binary label (above/below the curve)
#============================================
train_size          = 8000 # 80 800 8000
test_size           = 2000 # 20 200 2000
#
batch_size          = test_size // 2
hidden              = 32 # 32
dc                  = hidden // 8
num_hidden_layers   = 2
epochs              = 40
lr                  = 1e-3
n_proj_iters        = 2
#
ds_noise            = 0.05

def boundary_function(x1):

    #=============================
    a               = 0.0
    b               = 0.0
    c               = -1.
    sigma           = 1.0 # sharp 0.1
    high_freq       = 10
    high_freq_amp   = 0.1
    #=============================
    x0 = torch.tensor(math.pi, device=x1.device, dtype=x1.dtype)
    return a + b * x1 + c * torch.exp(-((x1 - x0) ** 2) / (sigma ** 2)) + high_freq_amp * torch.sin(high_freq * x1)
#
cfg = DotDict({
    "seed":                 0,
    "hidden":               hidden, # train_size // 4, # 64, # 32 # 256
    "num_hidden_layers":    num_hidden_layers,
    "epochs":               epochs,
    "lr":                   lr, # 1e-3, Interesting result with 1e-4
    "batch_size":           batch_size, # 256
    #
    "device": "cuda" if torch.cuda.is_available() else "cpu",
})
cfg.Dc = dc
#============================================

seed_everything(cfg.seed)

train_ds = WigglyCurveDataset(n=train_size,     seed=cfg.seed,      noise=ds_noise, boundary_fn=boundary_function)
# Shuffle train just once
g = torch.Generator().manual_seed(cfg.seed)
perm = torch.randperm(len(train_ds), generator=g)
train_ds_shuffled = torch.utils.data.Subset(train_ds, perm.tolist())

test_ds  = WigglyCurveDataset(n=test_size,      seed=cfg.seed + 1,  noise=ds_noise, boundary_fn=boundary_function)

test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False)

seed_everything(cfg.seed)
base_model_plane = ToyNet(hidden=cfg.hidden, Dc=cfg.Dc, num_hidden_layers=cfg.num_hidden_layers,
                          use_projector=False).to(cfg.device)
seed_everything(cfg.seed)
base_model_proj  = ToyNet(hidden=cfg.hidden, Dc=cfg.Dc, num_hidden_layers=cfg.num_hidden_layers,
                          use_projector=True, n_proj_iters=n_proj_iters).to(cfg.device)

print("INIT checksum plane:", checksum(base_model_plane))
print("INIT checksum proj :", checksum(base_model_proj))

train_loader = DataLoader(train_ds_shuffled, batch_size=cfg.batch_size, shuffle=False) # already shuffled

hist_plain = train(
    base_model_plane,
    # train_loader_plane,
    train_loader,
    test_loader,
    epochs=cfg.epochs,
    lr=cfg.lr,
    device=cfg.device
)

hist_proj = train(
    base_model_proj,
    # train_loader_proj,
    train_loader,
    test_loader,
    epochs=cfg.epochs,
    lr=cfg.lr,
    device=cfg.device
)
print_metrics(hist_plain, hist_proj)

plot_metrics_curves(
    hist_plain,
    hist_proj,
    title_fs=18,
    label_fs=14,
    tick_fs=12,
)

print("alpha_h (proj) =", torch.sigmoid(base_model_proj.logit_alpha_h).item())

plot_decision_boundaries_side_by_side(
    base_model_plane,
    base_model_proj,
    train_ds,
    device=cfg.device
)