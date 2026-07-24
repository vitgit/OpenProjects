import os
import random
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import Dataset
from torchvision.datasets import STL10

from ConvProjector import ResNetWithProjectorConv

print("STL10 Dataset")
'''
Each class in STL10 is represented by an integer [0-9]:
0: Airplane | 1: Automobile | 2: Bird | 3: Cat | 4: Deer
5: Dog | 6: Frog | 7: Horse | 8: Ship | 9: Truck
'''
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- dataset wrapper ----
class STLBinary(Dataset):
    def __init__(self, data, targets, indices, transform=None):
        self.data = data
        self.targets = [targets[i] for i in indices]
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x = self.data[self.indices[i]]
        y = self.targets[i]

        # x = torch.tensor(x).permute(2,0,1).float() / 255.0
        x = torch.tensor(x).float() / 255.0
        if self.transform:
            x = self.transform(x)

        return x, y

def prepare_data(data_root_folder, animal_indices, train_part=0.8):
    # ---- transform ----
    transform = transforms.ToTensor()

    # ---- load ----
    train_raw = STL10(root=data_root_folder, split='train', download=True, transform=transform)
    test_raw  = STL10(root=data_root_folder, split='test',  download=True, transform=transform)

    # ---- merge ----
    data = list(train_raw.data) + list(test_raw.data)
    labels = list(train_raw.labels) + list(test_raw.labels)

    # ---- animals vs vehicles ----
    targets = [1 if t in animal_indices else 0 for t in labels]

    # ---- balance ----
    idx_0 = [i for i, t in enumerate(targets) if t == 0]
    idx_1 = [i for i, t in enumerate(targets) if t == 1]

    n = min(len(idx_0), len(idx_1))

    rng = np.random.default_rng(42)
    idx_0 = rng.choice(idx_0, n, replace=False)
    idx_1 = rng.choice(idx_1, n, replace=False)

    indices = np.concatenate([idx_0, idx_1])
    rng.shuffle(indices)

    # ---- split 80/20 ----
    split = int(train_part * len(indices))
    train_idx = indices[:split]
    test_idx  = indices[split:]

    train_set = STLBinary(data, targets, train_idx)
    test_set  = STLBinary(data, targets, test_idx)

    return train_set, test_set

# === plot functions =============================================== Begin
def plot_metrics_curves(
        hist_plain,
        hist_proj,
        metrics_fixed_01=True,   # <<< NEW SWITCH
        title_fs=16,
        label_fs=13,
        tick_fs=11,
        out_pdf=None,
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

    metric_keys = {"acc", "prec", "rec", "f1"}

    for ax, (title, key) in zip(axes, plots):
        ax.plot(epochs, hist_plain[key], label="Plain", linewidth=2)
        ax.plot(epochs, hist_proj[key], label="Proj", linestyle="--", linewidth=2)

        ax.set_title(title, fontsize=title_fs)
        ax.set_xlabel("Epoch", fontsize=label_fs, labelpad=6)
        ax.set_ylabel(title, fontsize=label_fs, labelpad=6)

        # <<< only metrics get [0, 1] if enabled
        if metrics_fixed_01 and key in metric_keys:
            ax.set_ylim(0.0, 1.0)

        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(True)
        ax.legend(fontsize=label_fs - 1)

    fig.tight_layout()

    if out_pdf is not None:
        fig.savefig(out_pdf)

    plt.show()
    plt.close(fig)

def plot_metrics_curves_acc_f1(
        hist_plain,
        hist_proj,
        metrics_fixed_01=True,
        title_fs=16,
        label_fs=13,
        tick_fs=11,
        out_pdf=None,
):
    epochs = range(1, len(hist_plain["train_loss"]) + 1)

    # 3 rows × 2 cols -> one empty subplot
    fig, axes = plt.subplots(3, 2, figsize=(11, 14))
    axes = axes.ravel()

    plots = [
        ("Val Accuracy", "acc"),
        ("Val F1-score", "f1"),
        ("AUROC", "auc"),
        ("Train loss", "train_loss"),
        ("Validation loss", "val_loss"),
    ]

    metric_keys = {"acc", "f1", "auc"}

    for ax, (title, key) in zip(axes, plots):
        ax.plot(epochs, hist_plain[key], label="Plain", linewidth=2)
        ax.plot(epochs, hist_proj[key], label="Proj", linestyle="--", linewidth=2)

        ax.set_title(title, fontsize=title_fs)
        ax.set_xlabel("Epoch", fontsize=label_fs, labelpad=6)
        ax.set_ylabel(title, fontsize=label_fs, labelpad=6)

        if metrics_fixed_01 and key in metric_keys:
            ax.set_ylim(0.0, 1.0)

        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(True)
        ax.legend(fontsize=label_fs - 1)

    # hide last empty subplot
    axes[-1].axis("off")

    fig.tight_layout()

    if out_pdf is not None:
        fig.savefig(out_pdf)

    plt.show()
    plt.close(fig)

def plot_log_grad_norm(history_plain, history_proj, title_fs=16, out_pdf=None):
    g_plain = history_plain["grad_norm"]
    g_proj  = history_proj["grad_norm"]

    epochs = range(1, len(g_plain) + 1)

    # numerical safety
    eps = 1e-12
    g_plain = [np.log10(max(x, eps)) for x in g_plain]
    g_proj  = [np.log10(max(x, eps)) for x in g_proj]

    fig = plt.figure(figsize=(7, 5))
    plt.plot(epochs, g_plain, label="Plain", linewidth=2)
    plt.plot(epochs, g_proj,  label="Projector", linewidth=2, linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("log10(grad norm)")
    plt.title("Gradient Norm Dynamics (log scale)", fontsize=title_fs)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if out_pdf is not None:
        fig.tight_layout()
        fig.savefig(out_pdf)
    plt.close(fig)

def plot_mask(mask, radius, cutoff_low, cutoff_high):

    plt.figure(figsize=(6, 6))

    # show normalized radius field
    im = plt.imshow(
        radius.cpu().numpy(),
        cmap="viridis",
        origin="lower"
    )

    # draw boundary of retained region
    plt.contour(
        mask.cpu().numpy(),
        levels=[0.5],
        colors="red",
        linewidths=2
    )

    plt.title(
        f"Radial Frequency Mask\n"
        f"$r \\in [{cutoff_low}, {cutoff_high}]$"
    )

    plt.colorbar(im, label="Normalized Radius")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# === plot functions =============================================== End

def train_n_stats(model_plain, model_proj, train_loader, test_loader, epochs, device,
                  opt_plain, opt_proj, criterion, plot_picture_images_at_batch=True):

    stats_plain = StatLogger()
    stats_proj  = StatLogger()

    history = {
        "plain": {
            "acc": [], "prec": [], "rec": [], "f1": [], "auc": [],
            "train_loss": [], "val_loss": [],
            "grad_mean": [], "grad_std": [],
        },
        "proj": {
            "acc": [], "prec": [], "rec": [], "f1": [], "auc": [],
            "train_loss": [], "val_loss": [],
            "grad_mean": [], "grad_std": [],
        },
    }

    for epoch in range(epochs):
        # print(f"\n===== Epoch {epoch+1} =====")
        print(f"\n======= Epoch {epoch+1}/{epochs} =======")

        grad_mean_p_epoch = []
        grad_std_p_epoch = []
        grad_mean_r_epoch = []
        grad_std_r_epoch = []

        stats_plain.reset()
        stats_proj.reset()

        model_plain.train()
        model_proj.train()

        loss_plain = 0.0
        loss_proj  = 0.0

        TP_p = FP_p = TN_p = FN_p = 0
        TP_r = FP_r = TN_r = FN_r = 0

        for batch_num, (x_clean, x, y) in enumerate(train_loader):

            # ===== prepare batch ONCE =====
            x = x.to(device)
            y = y.float().unsqueeze(1).to(device)
            x_clean = x_clean.to(device)
            #-------------------------------------------
            # test: destroy training by shuffling labels
            # y = torch.randint(0, 2, y.shape, device=y.device).float()
            #-------------------------------------------

            # =========================
            # ---- PLAIN STEP ----
            # =========================
            opt_plain.zero_grad()
            logits_p = model_plain(x)
            loss_p = criterion(logits_p, y)
            loss_p.backward()

            mean_p, std_p = grad_stats(model_plain)
            grad_mean_p_epoch.append(mean_p)
            grad_std_p_epoch.append(std_p)

            log_gradients(model_plain, stats_plain)
            opt_plain.step()

            loss_plain += loss_p.item() * x.size(0)

            preds_p = (torch.sigmoid(logits_p).detach() > 0.5).float()
            TP_p += (preds_p * y).sum().item()
            TN_p += ((1 - preds_p) * (1 - y)).sum().item()
            FP_p += (preds_p * (1 - y)).sum().item()
            FN_p += ((1 - preds_p) * y).sum().item()

            # =========================
            # ---- PROJ STEP ----
            # =========================
            opt_proj.zero_grad()
            #---------------------------------
            if batch_num == 1:
                model_proj.plot_image = plot_picture_images_at_batch
            #---------------------------------
            model_proj.x_clean = x_clean
            logits_r = model_proj(x)
            #---------------------------------
            model_proj.plot_image = False
            #---------------------------------
            loss_r = criterion(logits_r, y)
            loss_r.backward()

            mean_r, std_r = grad_stats(model_proj)
            grad_mean_r_epoch.append(mean_r)
            grad_std_r_epoch.append(std_r)

            log_gradients(model_proj, stats_proj)
            opt_proj.step()

            loss_proj += loss_r.item() * x.size(0)

            preds_r = (torch.sigmoid(logits_r).detach() > 0.5).float()
            TP_r += (preds_r * y).sum().item()
            TN_r += ((1 - preds_r) * (1 - y)).sum().item()
            FP_r += (preds_r * (1 - y)).sum().item()
            FN_r += ((1 - preds_r) * y).sum().item()

        loss_plain /= len(train_loader.dataset)
        loss_proj  /= len(train_loader.dataset)

        val_loss_p, acc_pv, prec_pv, rec_pv, f1_pv, auc_pv = evaluate(model_plain, test_loader, criterion)
        val_loss_r, acc_rv, prec_rv, rec_rv, f1_rv, auc_rv = evaluate(model_proj, test_loader, criterion)

        print("\n--- PLAIN vs PROJECTOR ---")

        print(f"{'Metric':<15} | {'Plain':>10} | {'Proj':>10}")
        print("-" * 40)

        print(f"{'Accuracy':<15} | {acc_pv:>10.4f} | {acc_rv:>10.4f}")
        print(f"{'Precision':<15} | {prec_pv:>10.4f} | {prec_rv:>10.4f}")
        print(f"{'Recall':<15} | {rec_pv:>10.4f} | {rec_rv:>10.4f}")
        print(f"{'F1-score':<15} | {f1_pv:>10.4f} | {f1_rv:>10.4f}")
        print(f"{'AUROC':<15} | {auc_pv:>10.4f} | {auc_rv:>10.4f}")

        print("-" * 40)
        print(f"{'Train Loss':<15} | {loss_plain:>10.4f} | {loss_proj:>10.4f}")
        print(f"{'Val Loss':<15} | {val_loss_p:>10.4f} | {val_loss_r:>10.4f}")

        gn_plain = stats_plain.summary().get("grad_norm")
        gn_proj = stats_proj.summary().get("grad_norm")

        history["plain"].setdefault("grad_norm", []).append(gn_plain)
        history["proj"].setdefault("grad_norm", []).append(gn_proj)

        # ---- plain ----
        history["plain"]["train_loss"].append(loss_plain)
        history["plain"]["val_loss"].append(val_loss_p)
        history["plain"]["acc"].append(acc_pv)
        history["plain"]["prec"].append(prec_pv)
        history["plain"]["rec"].append(rec_pv)
        history["plain"]["f1"].append(f1_pv)
        history["plain"]["auc"].append(auc_pv)
        history["plain"]["grad_mean"].append(np.mean(grad_mean_p_epoch))
        history["plain"]["grad_std"].append(np.mean(grad_std_p_epoch))

        # ---- proj ----
        history["proj"]["train_loss"].append(loss_proj)
        history["proj"]["val_loss"].append(val_loss_r)
        history["proj"]["acc"].append(acc_rv)
        history["proj"]["prec"].append(prec_rv)
        history["proj"]["rec"].append(rec_rv)
        history["proj"]["f1"].append(f1_rv)
        history["proj"]["auc"].append(auc_rv)
        history["proj"]["grad_mean"].append(np.mean(grad_mean_r_epoch))
        history["proj"]["grad_std"].append(np.mean(grad_std_r_epoch))

        print("\n--- Gradient stats ---")
        print(f"{'Metric':<15} | {'Plain':>10} | {'Proj':>10}")
        print("-" * 40)
        print(f"{'Grad mean':<15} | {history['plain']['grad_mean'][-1]:>10.4f} | {history['proj']['grad_mean'][-1]:>10.4f}")
        print(f"{'Grad std':<15} | {history['plain']['grad_std'][-1]:>10.4f} | {history['proj']['grad_std'][-1]:>10.4f}")

        print(f'alpha: {model_proj.current_alpha}')
        print(f'alpha gradient: {model_proj.logit_alpha.grad}')

        plot_metrics_curves(
            history["plain"],
            history["proj"],
            metrics_fixed_01=True,
            # out_pdf=pdf_file,
        )

        plot_metrics_curves(
            history["plain"],
            history["proj"],
            metrics_fixed_01=False,
            # out_pdf=pdf_file,
        )

        plot_metrics_curves_acc_f1(
            history["plain"],
            history["proj"],
            metrics_fixed_01=False,
            # out_pdf=pdf_file,
        )

        plot_log_grad_norm(
            history["plain"],
            history["proj"],
        )

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0
    total_samples = 0
    TP = FP = TN = FN = 0

    all_probs = []
    all_targets = []

    for batch in loader:
        if len(batch) == 3:
            x_clean, x, y = batch
        else:
            x, y = batch
        x = x.to(device)
        y = y.float().unsqueeze(1).to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

        probs = torch.sigmoid(logits)

        preds = (torch.sigmoid(logits) > 0.5).float()

        TP += ((preds == 1) & (y == 1)).sum().item()
        TN += ((preds == 0) & (y == 0)).sum().item()
        FP += ((preds == 1) & (y == 0)).sum().item()
        FN += ((preds == 0) & (y == 1)).sum().item()

        all_probs.append(probs.cpu())
        all_targets.append(y.cpu())

    total_loss /= total_samples
    acc, prec, rec, f1 = metrics(TP, TN, FP, FN)

    # ---- AUROC ----
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    roc_auc = roc_auc_score(all_targets, all_probs)

    return total_loss, acc, prec, rec, f1, roc_auc

# ---- compute train metrics ----
def metrics(TP, TN, FP, FN):
    acc  = (TP + TN) / max(TP + TN + FP + FN, 1)
    prec = TP / max(TP + FP, 1)
    rec  = TP / max(TP + FN, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-12)
    return acc, prec, rec, f1

def grad_stats(model):
    norms = []
    for p in model.parameters():
        if p.grad is not None:
            norms.append(p.grad.norm().item())
    return np.mean(norms), np.std(norms)

class StatLogger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = defaultdict(list)

    def log(self, key, value):
        if value is None:
            return
        if torch.is_tensor(value):
            value = value.detach().float().cpu().item()
        self.data[key].append(value)

    def summary(self):
        return {k: sum(v) / len(v) for k, v in self.data.items() if len(v) > 0}

def log_gradients(model, stats):
    total_sq = 0.0

    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.pow(2).sum().item()

    global_norm = total_sq ** 0.5
    stats.log("grad_norm", global_norm)

# === Distortion =============================================== Begin

class NoisyDataset(Dataset):
    def __init__(
            self,
            base_dataset,
            noise_type=None,     # "gaussian" or "high_freq"
            noise_std=0.0,
            cutoff_high=0.2,
            cutoff_low=0.0,
            noise_prob=1.0
    ):
        self.base = base_dataset
        self.noise_type = noise_type
        self.std = noise_std
        self.cutoff_high = cutoff_high
        self.cutoff_low = cutoff_low
        self.noise_prob = noise_prob
        self.targets = base_dataset.targets
        self.grid_cache = {}

        # ---- precompute once ----
        self.samples = []

        for idx in range(len(self.base)):

            x_clean, y = self.base[idx]

            x = x_clean.clone()

            if self.noise_type is not None and self.std > 0:

                if torch.rand(()) < self.noise_prob:

                    if self.noise_type == "gaussian":
                        x = self.add_gaussian_noise(x)

                    elif self.noise_type == "high_freq":
                        x = self.add_high_freq_noise(x)

            self.samples.append((x_clean, x, y))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return self.samples[idx]

    def add_gaussian_noise(self, x):
        x = x + self.std * torch.randn_like(x)
        return torch.clamp(x, 0.0, 1.0)

    def add_high_freq_noise(self, x):
        if self.std == 0:
            return x

        C, H, W = x.shape

        # === FFT of image ===
        X = torch.fft.fft2(x)
        X = torch.fft.fftshift(X)

        # === build / reuse radius grid ===
        if (H, W) not in self.grid_cache:
            y = torch.linspace(-1, 1, H, dtype=torch.float32)
            x_ = torch.linspace(-1, 1, W, dtype=torch.float32)
            yy, xx = torch.meshgrid(y, x_, indexing='ij')
            radius = torch.sqrt(xx**2 + yy**2)
            self.grid_cache[(H, W)] = radius

        radius = self.grid_cache[(H, W)].to(x.device)
        radius = radius / radius.max()

        # === high-frequency mask ===
        high_mask = ((radius >= self.cutoff_low) & (radius <= self.cutoff_high) ).to(X.real.dtype).unsqueeze(0)


        if not hasattr(self, "_plotted"):
            # plot_mask(high_mask.squeeze(0), self.cutoff_low, self.cutoff_high)
            plot_mask(high_mask.squeeze(0), radius, self.cutoff_low, self.cutoff_high)
            self._plotted = True

        # === generate noise in IMAGE space ===
        noise_img = torch.randn_like(x)

        # FFT of noise
        N = torch.fft.fft2(noise_img)
        N = torch.fft.fftshift(N)

        # keep ONLY high frequencies
        N = N * high_mask

        # add to signal
        X = X + N

        # === back to image ===
        X = torch.fft.ifftshift(X)
        x_noisy = torch.fft.ifft2(X).real

        # === normalize perturbation ===
        delta = x_noisy - x
        delta = delta / (delta.std() + 1e-8)
        delta = self.std * delta
        delta = delta - delta.mean()

        x_noisy = x + delta

        return torch.clamp(x_noisy, 0.0, 1.0)

def show_pair(x, x_noisy):
    x = x.permute(1, 2, 0).cpu().numpy()
    x_noisy = x_noisy.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))

    ax[0].imshow(x)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(x_noisy)
    ax[1].set_title("Noisy")
    ax[1].axis("off")

    plt.show()

def show_difference(x, x_noisy):
    diff = (x_noisy - x).abs()
    diff = diff / (diff.max() + 1e-8)

    plt.imshow(diff.permute(1,2,0).cpu().numpy())
    plt.title("Normalized difference")
    plt.axis("off")
    plt.show()

# === Distortion =============================================== End

class Logger:
    def __init__(self, *files):
        self.files = files
    def write(self, msg):
        for f in self.files:
            f.write(msg)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def _before():
    return list(globals().keys())  # preserve order

def _after(_before):
    before_set = set(_before)
    new_vars = ['\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>']
    now = datetime.now()
    new_vars.append(str(now))

    for var in globals():  # natural order (insertion order)
        if var not in before_set:
            # print(f"{var} = {globals()[var]}")
            new_vars.append(f'{var} = {globals()[var]}')
    return new_vars

def name_now():
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return current_time

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

base_dir = os.path.dirname(__file__)
_before = _before()
#============================================================
data_root_folder                = f'{base_dir}/data/stl10'
train_part                      = 0.8
#
batch_size                      = 64 # 64
lr                              = 1e-3  # 1e-3 1e-2 1e-4
epochs                          = 30 # 10
#
projector_name                  = 'ConvProjectorExactInverse' # 'ConvProjectorQQinv' 'ConvProjectorExactInverse'
alpha_const                     = None # 0.2 # 0.5 if 1.0 plain and proj identical # None - learnable
########
Dc                              = 3 # 16 # 3
# or
stride                          = 2
########
kernel_size_Q                   = 3
kernel_size_QQ_inv              = 3 # 1 3 5 9 # looks like 3 is optimal
#
plot_picture_images_at_batch    = True
#
noise_type                      = "high_freq" # "high_freq" "gaussian"
noise_train_std                 = 6.0 # 0.001 # 0.2 0.5 1.0 # 2.0 3.0 4.0 5.0
cutoff_train                    = (0.8, 1.0) # (0.9, 1.0) # (0.95, 1.0) # (0.8, 1.0) # (0.6, 1.0)
noise_train_prob                = 1.0 # 0.8 # 0.5
#
noisy_test                      = False # False True
noise_test_std                  = noise_train_std # 0.1 # 0.02 # 0.5 # 0.05
cutoff_test                     = cutoff_train # (0.0, 1.0) # (0.2, 0.6) # 0.05 # 0.2
noise_test_prob                 = noise_train_prob # 1.0 # 0.5
'''
noise_prob = 1.0 → every image is noisy
noise_prob = 0.5 → half noisy, half clean
noise_prob = 0.0 → no noise at all
'''
#============================================================
# >>> loging >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
printout_file_name  = f'{name_now()}_printout.log' # None
printout_file       = f'{base_dir}/data/stl10/logs/{printout_file_name}'
_after(_before)
new_vars = _after(_before)
# Activate printout logging
if printout_file_name is not None:
    log_file = open(printout_file, "a")
    sys.stdout = Logger(sys.stdout, log_file)
# print input parameters
for var in new_vars:
    print(var)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

animal_indices = {1, 3, 4, 5, 6, 7}

train_base, test_set = prepare_data(data_root_folder, animal_indices, train_part=train_part)

print('\nAfter balance:')
num_1 = sum(train_base.targets)
num_0 = len(train_base.targets) - num_1
print(f'train: {len(train_base)}, classes: ({num_0}, {num_1})')
num_1 = sum(test_set.targets)
num_0 = len(test_set.targets) - num_1
print(f'test: {len(test_set)}, classes: ({num_0}, {num_1})')
print()

if noise_train_std > 0.0:
    train_set = NoisyDataset(
        train_base,
        noise_type=noise_type,
        noise_std=noise_train_std,
        cutoff_low=cutoff_train[0],
        cutoff_high=cutoff_train[1],
        noise_prob=noise_train_prob
    )
else:
    train_set = train_base

x, y = train_base[0]
torch.manual_seed(0)
x_clean, x_noisy, y = train_set[0]

print(f'Noise level:\n{(x_noisy - x_clean).abs().mean()}')

print()
print(f'x_clean.shape = {x_clean.shape}')
print(f'x_noisy.shape = {x_noisy.shape}')
C, H, W = x_clean.shape
print(f'Image resolution: {H} x {W}')
print(f'Number of channels: {C}')

print()

show_pair(x_clean, x_noisy)

show_difference(x_clean, x_noisy)

if noise_test_std > 0.0:
    if noisy_test:
        test_set = NoisyDataset(
            test_set,
            noise_type=noise_type,
            noise_std=noise_test_std,
            cutoff_low=cutoff_test[0],
            cutoff_high=cutoff_test[1],
            noise_prob=noise_test_prob
        )
        print('Noise added to test set')

g = torch.Generator()
g.manual_seed(seed)
test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    generator=g
)
g = torch.Generator()
g.manual_seed(seed)
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    generator=g
)

# 4. Verification
print(f"Training images: {len(train_set)} (Animals: {train_set.targets.count(1)}, Vehicles: {train_set.targets.count(0)})")
print(f"Test images:     {len(test_set)} (Animals: {test_set.targets.count(1)}, Vehicles: {test_set.targets.count(0)})")

torch.manual_seed(seed)
model_plain = ResNetWithProjectorConv(use_projector=False, alpha_const=alpha_const).to(device)

torch.manual_seed(seed)
model_proj = ResNetWithProjectorConv(
                                     projector_name=projector_name,
                                     Dc=Dc,
                                     stride=stride,
                                     use_projector=True,
                                     alpha_const=alpha_const,
                                     kernel_size_Q=kernel_size_Q,
                                     kernel_size_QQ_inv=kernel_size_QQ_inv
                                     ).to(device)

print('\nBelow both should be True. Both models start from identical initial conditions:')
print(torch.allclose(model_plain.classifier.weight, model_proj.classifier.weight))
print(torch.allclose(model_plain.classifier.bias, model_proj.classifier.bias))

opt_plain = torch.optim.Adam(model_plain.parameters(), lr=lr)
opt_proj  = torch.optim.Adam(model_proj.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

train_n_stats(model_plain, model_proj, train_loader, test_loader, epochs, device,
              opt_plain, opt_proj, criterion,
              plot_picture_images_at_batch=plot_picture_images_at_batch)
