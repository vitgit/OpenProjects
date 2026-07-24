import json
import os
import random
import shutil
import sys
import warnings
from collections import Counter
from datetime import datetime
from typing import Tuple
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, set_seed

TextSample = Tuple[str, int]   # (text, label)

def load_all_data(jsonl_paths):
    data = []
    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    return data


def balance_data(data, seed=42):

    g = torch.Generator().manual_seed(seed)
    by_class = {}
    for ex in data:
        by_class.setdefault(ex["label"], []).append(ex)

    min_count = min(len(v) for v in by_class.values())

    balanced = []
    for v in by_class.values():
        # idx = torch.randperm(len(v))[:min_count]
        idx = torch.randperm(len(v), generator=g)[:min_count]
        balanced.extend([v[i] for i in idx])
    return balanced

class TextBinaryDataset(Dataset):
    def __init__(self, samples: list[TextSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_text(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels, dtype=torch.long)

class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-uncased", max_len=64):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.eval()  # frozen
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.max_len = max_len

    @torch.no_grad()
    def forward(self, texts: list[str]):
        batch = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        device = next(self.encoder.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = self.encoder(**batch)
        return outputs.last_hidden_state

# ==== Model =========================================== Begin

class TinyTransformer(nn.Module):
    def __init__(
            self,
            D,
            Dc_list = None,
            T=None,
            Tc=None,
            num_classes=2,
            projector_type=None,
            use_MLP=True,
            num_smoothing_steps=2,
            num_refinement_steps=1,
            proj_once=True,
            normalize_proj=False,
            projector_location="hidden",
            eta_constant=None,
    ):
        super().__init__()

        # NOTE:
        # Projector is non-idempotent. To avoid representation collapse,
        # it is applied at most once per refinement step.

        if Dc_list is None:
            Dc_list = [8, 16, 32]
        self.Dc_list = Dc_list

        if projector_type is None:
            num_smoothing_steps = 0
        if num_smoothing_steps is None or num_smoothing_steps < 1:
            projector_type = None
        if projector_type is None:
            projector_location = None

        if projector_type and num_refinement_steps > 1 and num_smoothing_steps > 1:
            warnings.warn(
                "Multiple smoothing steps per refinement increase contraction strength. "
                "Ensure alpha is not too small."
            )

        self.use_MLP                = use_MLP
        self.num_smoothing_steps    = num_smoothing_steps
        self.projector_type         = projector_type
        self.num_refinement_steps   = num_refinement_steps
        self.proj_once              = proj_once
        self.normalize_proj         = normalize_proj
        self.projector_location     = projector_location
        self.eta_constant           = eta_constant

        # --- Transformer blocks (duplicated) ---
        self.attn_blocks = nn.ModuleList([
            nn.MultiheadAttention(D, num_heads=4, batch_first=True)
            for _ in range(num_refinement_steps)
        ])

        self.norm1 = nn.ModuleList([nn.LayerNorm(D) for _ in range(num_refinement_steps)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(D) for _ in range(num_refinement_steps)])

        if self.use_MLP:
            self.mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(D, 4 * D),
                    nn.GELU(),
                    nn.Linear(4 * D, D)
                )
                for _ in range(num_refinement_steps)
            ])
        else:
            self.mlps = None

        # --- Shared projector ---
        if projector_type is not None:
            self.projs = nn.ModuleList([
                Projector(D, Dc_i, T, Tc, projector_type)
                for Dc_i in Dc_list
            ])

            self.alpha_logits = nn.Parameter(torch.zeros(len(Dc_list)))
            self.logit_eta_h = nn.Parameter(torch.tensor(0.0))

        else:
            self.projs = None

        self.fc = nn.Linear(D, num_classes)

    def apply_projector(self, h, stats=None, step_idx=None, normalize_proj=False):
        if self.projs is None:
            return h

        # convex weights
        alpha = torch.softmax(self.alpha_logits, dim=0)   # [K]

        if self.eta_constant is not None:
            # eta = self.eta_constant
            eta = torch.as_tensor(
                self.eta_constant,
                device=h.device,
                dtype=h.dtype
            )
        else:
            eta = torch.sigmoid(self.logit_eta_h)

        if stats is not None:
            for i, a in enumerate(alpha):
                stats.log(f"alpha_{i}", a)
            stats.log("eta", eta)

        for k in range(self.num_smoothing_steps):
            h_before = h
            # mixed coarse correction
            h_proj = torch.zeros_like(h)

            if normalize_proj:
                h_norm = h.norm(dim=-1, keepdim=True) + 1e-6  # precompute once
                for a, proj in zip(alpha, self.projs):
                    proj_h = proj(h)
                    proj_norm = proj_h.norm(dim=-1, keepdim=True) + 1e-6
                    scale = (h_norm / proj_norm).detach()
                    proj_h = proj_h * scale
                    h_proj = h_proj + a * proj_h               # <-- mix
            else:
                for a, proj in zip(alpha, self.projs):
                    h_proj = h_proj + a * proj(h)
            h = h + eta * (h_proj - h)

            if stats is not None:
                delta = (h_before - h).norm(dim=-1).mean()
                ratio = delta / (h_before.norm(dim=-1).mean() + 1e-8)

                stats.log(f"proj_delta_step{step_idx}", delta)
                stats.log(f"proj_ratio_step{step_idx}", ratio)

        return h

    def forward(self, x, stats=None):
        h = x
        if self.projector_location == "input":
            h = self.apply_projector(
                h,
                stats=stats,
                step_idx="input",
                normalize_proj=self.normalize_proj
            )
        for ii in range(self.num_refinement_steps):
            # --- Attention ---
            h_norm = self.norm1[ii](h)
            h_attn, _ = self.attn_blocks[ii](h_norm, h_norm, h_norm)
            h = h + h_attn

            # --- MLP ---
            if self.use_MLP:
                h_norm = self.norm2[ii](h)
                h = h + self.mlps[ii](h_norm)

            # --- Log before projector ---
            if stats is not None:
                stats.log(f"smooth_t_pre_step{ii}", temporal_smoothness(h))
                stats.log(f"spectral_pre_step{ii}", spectral_tail(h))

            # --- Projector after EACH block ---
            if self.projector_location == "hidden":
                if self.proj_once and ii == self.num_refinement_steps - 1:
                    h = self.apply_projector(
                        h,
                        stats=stats,
                        step_idx=ii,
                        normalize_proj=self.normalize_proj)

            # --- Log after projector ---
            if stats is not None:
                stats.log(f"smooth_t_post_step{ii}", temporal_smoothness(h))
                stats.log(f"spectral_post_step{ii}", spectral_tail(h))

        # Pool over sequence
        h = h.mean(dim=1)   # [B, D]
        return self.fc(h)

class Projector(nn.Module):
    def __init__(self, D, Dc, T, Tc, projector_type="both", eps=1e-6):
        super().__init__()

        self.projector_type = projector_type
        self.eps = eps

        if projector_type in ["feature", "both"]:
            self.Qs = Restrict(D, Dc)
            self.Q  = Prolong(Dc, D)

        if projector_type in ["sequence", "both"]:
            self.seq_proj = SequenceProjector(T, Tc)

    def forward(self, x):
        # x: [B, T, D]

        if self.projector_type in ["sequence", "both"]:
            x = self.seq_proj(x)

        if self.projector_type in ["feature", "both"]:
            z = self.Qs(x) # [B, T, Dc]

            Wq  = self.Q.proj.weight
            Wqs = self.Qs.proj.weight
            A = Wqs @ Wq
            A = A + self.eps * torch.eye(A.size(0), device=A.device)

            B, T, Dc = z.shape
            z_flat = z.reshape(-1, Dc)
            u = torch.linalg.solve(A, z_flat.T)
            u = u.T.reshape(B, T, Dc)

            x = self.Q(u)

            if not hasattr(self, "A_shape"):
                self.A_shape = A.shape           # (Dc, Dc)
                self.RHS_shape = z_flat.T.shape  # (Dc, B*T)

        return x

class Restrict(nn.Module):  # Q*
    def __init__(self, D, Dc):
        super().__init__()
        self.proj = nn.Linear(D, Dc, bias=False)

    def forward(self, x):
        # x: [B, T, D] → [B, T, Dc]
        return self.proj(x)

class Prolong(nn.Module):  # Q
    def __init__(self, Dc, D):
        super().__init__()
        self.proj = nn.Linear(Dc, D, bias=False)

    def forward(self, z):
        # z: [B, T, Dc] → [B, T, D]
        return self.proj(z)

class SequenceProjector(nn.Module):
    def __init__(self, T, Tc):
        super().__init__()
        # self.W = nn.Parameter(torch.randn(T, Tc))
        self.W = nn.Parameter(0.02 * torch.randn(T, Tc))

    def forward(self, x):
        """
        x: [B, T, D]
        """
        assert x.size(1) == self.W.size(0), f"T mismatch: x.T={x.size(1)} vs W.T={self.W.size(0)}"
        # Orthonormal basis in time with QR decomposition, so no inversion needed
        Q, _ = torch.linalg.qr(self.W, mode="reduced")  # [T, Tc]
        Q = torch.sign(Q[0:1, :]) * Q
        Pt = Q @ Q.T              # [T, T]
        return Pt.unsqueeze(0) @ x  # [1,T,T] @ [B,T,D] -> [B,T,D]

def eta_schedule(epoch, max_epochs, a0=0.2, a1=1.0):
    return a1 - (a1 - a0) * epoch / max_epochs
# ==== Model =========================================== End

def print_data_diagnostics(
        train_or_test,
        dataset,
        dataloader,
        encoder,
        D,
        Dc_list,
        device,
):
    print(f"\n====== Data diagnostics for {train_or_test} set ===============")
    print(f"Dataset size            : {len(dataset)}")
    print(f"Batch size              : {dataloader.batch_size}")
    print(f"Number of batches       : {len(dataloader)}")
    print(f"Embedding dim (D)       : {D}")
    print(f"Coarse dims (Dc_list)   : {Dc_list}")
    print(f"Device                  : {device}")

    # Inspect one batch
    texts, y = next(iter(dataloader))

    print("\nBatch-level:")
    print(f"  Text count        : {len(texts)}")
    print(f"  Labels shape      : {y.shape}")
    print(f"  Labels dtype      : {y.dtype}")
    print(f"  Example text[0]   : {texts[0][:120]}{'...' if len(texts[0]) > 120 else ''}")

    encoder.eval()
    with torch.no_grad():
        X = encoder(texts)

    print("\nEncoded tensor:")
    print(f"  X shape [B, T, D] : {tuple(X.shape)}")
    print(f"  X dtype           : {X.dtype}")
    print(f"  X device          : {X.device}")

    # ---- Token length diagnostics (C) ----
    tokenized = encoder.tokenizer(
        texts,
        truncation=True,
        max_length=encoder.max_len,
        return_length=True,
    )
    lengths = tokenized["length"]
    lengths = torch.tensor(lengths)
    print("\nToken length stats:")
    print(f"  Mean tokens       : {lengths.float().mean():.1f}")
    print(f"  Max tokens        : {lengths.max().item()}")
    print(f"  % truncated       : {(lengths == encoder.max_len).float().mean() * 100:.2f}%")
    print("===================================================\n")

def print_label_balance(name, dataset):
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)
    total = sum(counts.values())

    balance_dict = {}

    print(f"\n{name} label balance:")
    for cls in sorted(counts):
        n = counts[cls]
        pct = 100.0 * n / total
        val = f"  class {cls}: {n:6d} ({pct:6.2f}%)"
        print(val)
        balance_dict[cls] = val
    return balance_dict


def train_dual(
        model_plain,
        model_proj,
        encoder,
        train_loader,
        val_loader,
        epochs=10,
        lr=None,
        lr_change=None,
        lr_step_epoch=None,
        device="cpu",
        use_alpha_h_optimizer=False,
        alpha_h_lr=1e-4,
        lambda_entropy=1e-3,
        class_weights=None,
        images_dir=None,
        use_eta_schedule=False
):
    def set_lr(opt, lr_main):
        opt.param_groups[0]["lr"] = lr_main

    if lr is None:
        lr = [1e-3, 1e-3]

    model_plain.to(device)
    model_proj.to(device)
    encoder.to(device)
    encoder.eval()

    # ---------- Optimizers ----------
    opt_plain = torch.optim.Adam(model_plain.parameters(), lr=lr[0])

    if use_alpha_h_optimizer and model_proj.projector_type:
        if model_proj.eta_constant is None:
            alpha_params = [
                    model_proj.alpha_logits,
                    model_proj.logit_eta_h,
            ]
        else:
            alpha_params = [
                model_proj.alpha_logits,
            ]
        other_params = [
            p for n, p in model_proj.named_parameters()
            if n not in {"alpha_logits", "logit_eta_h"}
        ]


        opt_proj = torch.optim.Adam(
            [
                {"params": other_params, "lr": lr[0]},
                {"params": alpha_params, "lr": alpha_h_lr},
            ]
        )
    else:
        opt_proj = torch.optim.Adam(model_proj.parameters(), lr=lr[0])

    history = {
        "plain": {
            "acc": [], "prec": [], "rec": [], "f1": [], "auroc": [],
            "train_loss": [], "val_loss": [],
        },
        "proj": {
            "acc": [], "prec": [], "rec": [], "f1": [], "auroc": [],
            "train_loss": [], "val_loss": [],
        },
    }

    stats_plain = StatLogger()
    stats_proj  = StatLogger()


    for epoch in range(epochs):

        stats_plain.reset()
        stats_proj.reset()

        print(f"\n======= Epoch {epoch+1}/{epochs} =======")

        # ---------- LR schedule ----------
        lr1, lr2 = lr
        if lr_change == "linear":
            lr_main = lr1 + (lr2 - lr1) * epoch / (epochs - 1)
        elif lr_change == "step":
            lr_main = lr1 if epoch + 1 <= lr_step_epoch else lr2
        else:
            raise ValueError(f"Unknown lr_change: {lr_change}")

        set_lr(opt_plain, lr_main)
        set_lr(opt_proj,  lr_main)

        # ---------- eta schedule ----------
        if use_eta_schedule and model_proj.projector_type and model_proj.eta_constant is None:
            with torch.no_grad():
                a = eta_schedule(epoch+1, epochs)
                a = float(np.clip(a, 1e-6, 1 - 1e-6))
                model_proj.logit_eta_h.copy_(
                    torch.logit(torch.tensor(a, device=device))
                )

        model_plain.train()
        model_proj.train()

        loss_plain = 0.0
        loss_proj  = 0.0

        epoch_alpha_grad = 0.0
        # ---------- Training ----------
        for texts, y in train_loader:
            y = y.to(device)

            with torch.no_grad():
                X = encoder(texts)

            # ---- plain ----
            opt_plain.zero_grad()
            logits_p = model_plain(X)
            # lp = F.cross_entropy(logits_p, y)
            lp = F.cross_entropy(logits_p, y, weight=class_weights)
            lp.backward()

            # ===== gradient logging (PLAIN) =====
            log_gradients(model_plain, stats_plain)

            opt_plain.step()
            loss_plain += lp.item()

            # ---- projector ----
            opt_proj.zero_grad()
            logits_r = model_proj(X)
            # lr_ = F.cross_entropy(logits_r, y, weight=class_weights)
            # lr_.backward()
            lr_ = F.cross_entropy(logits_r, y, weight=class_weights)

            if model_proj.projector_type is not None:
                # --- alpha entropy regularization ---
                alpha = torch.softmax(model_proj.alpha_logits, dim=0)
                entropy = -(alpha * torch.log(alpha + 1e-8)).sum()
            else:
                entropy = 0.0

            lr_total = lr_ + lambda_entropy * entropy
            lr_total.backward()

            # print('\n',"alpha_logits.grad =", model_proj.alpha_logits.grad,'\n')
            if model_proj.projector_type is not None:
                epoch_alpha_grad += model_proj.alpha_logits.grad.detach()

            # ===== gradient logging (PROJ) =====
            log_gradients(model_proj, stats_proj)

            opt_proj.step()

            with torch.no_grad():
                if model_proj.projector_type is not None:
                    model_proj.alpha_logits.clamp_(-6.0, 6.0)
                    model_proj.logit_eta_h.clamp_(-6.0, 6.0)

            loss_proj += lr_.item()

        loss_plain /= len(train_loader)
        loss_proj  /= len(train_loader)

        # ---------- Validation ----------
        val_loss_p, acc_p, prec_p, rec_p, f1_p, auroc_p, _ = evaluate(
            model_plain, encoder, val_loader, device
        )
        val_loss_r, acc_r, prec_r, rec_r, f1_r, auroc_r, _ = evaluate(
            model_proj, encoder, val_loader, device
        )

        print()
        # print("STAT KEYS (plain):", stats_plain.data.keys())
        # print("STAT KEYS (proj) :", stats_proj.data.keys())

        print("\n--- Gradient diagnostics ---")
        print("Plain grad norm :", stats_plain.summary().get("grad_norm"))
        print("Proj  grad norm :", stats_proj.summary().get("grad_norm"))
        print()

        gn_plain = stats_plain.summary().get("grad_norm")
        gn_proj  = stats_proj.summary().get("grad_norm")
        history["plain"].setdefault("grad_norm", []).append(gn_plain)
        history["proj"].setdefault("grad_norm", []).append(gn_proj)

        # ---- plain ----
        history["plain"]["train_loss"].append(loss_plain)
        history["plain"]["val_loss"].append(val_loss_p)
        history["plain"]["acc"].append(acc_p)
        history["plain"]["prec"].append(prec_p)
        history["plain"]["rec"].append(rec_p)
        history["plain"]["f1"].append(f1_p)
        history["plain"]["auroc"].append(auroc_p)

        # ---- proj ----
        history["proj"]["train_loss"].append(loss_proj)
        history["proj"]["val_loss"].append(val_loss_r)
        history["proj"]["acc"].append(acc_r)
        history["proj"]["prec"].append(prec_r)
        history["proj"]["rec"].append(rec_r)
        history["proj"]["f1"].append(f1_r)
        history["proj"]["auroc"].append(auroc_r)

        print()

        print_metrics(acc_p, prec_p, rec_p, f1_p, auroc_p, loss_plain,   val_loss_p,
                      acc_r, prec_r, rec_r, f1_r, auroc_r, loss_proj,    val_loss_r)

        print(f"\nLR = {lr_main:.6f}")

        if model_proj.projector_type is not None:
            if model_proj.eta_constant is not None:
                eta = model_proj.eta_constant
            else:
                with torch.no_grad():
                    eta = torch.sigmoid(model_proj.logit_eta_h).item()
            print(f"eta : {eta:.4f}")
            if model_proj.eta_constant is None:
                print("eta grad:", model_proj.logit_eta_h.grad)
            alpha = torch.softmax(model_proj.alpha_logits, dim=0)
            print("alpha :", [f"{a:.3f}" for a in alpha.tolist()])

            print("mean alpha grad:", epoch_alpha_grad / len(train_loader))

        plot_metrics_curves(
            history["plain"],
            history["proj"],
            metrics_fixed_01=True,
            out_pdf=f'{images_dir}/metrics_01_{epoch}.pdf',
        )
        plot_metrics_curves(
            history["plain"],
            history["proj"],
            metrics_fixed_01=False,
            out_pdf=f'{images_dir}/metrics_minmax_{epoch}.pdf',
        )
        plot_log_grad_norm(
            history["plain"],
            history["proj"],
            out_pdf=f'{images_dir}/grad_norm_{epoch}.pdf',
        )
    return history

def print_lrs(opt, model, use_alpha_h_optimizer):

    # main LR
    lr_main = opt.param_groups[0]["lr"]

    # alpha_h LR (only if present)
    if use_alpha_h_optimizer and model.projector_type:
        lr_alpha = opt.param_groups[1]["lr"]
    else:
        lr_alpha = None

    print()
    print(f"lr_main = {lr_main:.6f}", end="")
    if lr_alpha is not None:
        print(f", lr_alpha = {lr_alpha:.6f}")
    else:
        print()

@torch.no_grad()
def evaluate(model, encoder, dataloader, device="cpu"):
    model.eval()
    encoder.eval()

    all_preds   = []
    all_labels  = []
    all_probs   = []
    total_loss  = 0.0

    for texts, y in dataloader:
        y = y.to(device)
        X = encoder(texts)
        logits = model(X)

        total_loss += F.cross_entropy(logits, y).item()

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.append(probs.cpu())

    total_loss /= len(dataloader)

    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_labels)
    y_prob = torch.cat(all_probs)

    acc, prec, rec, f1, auroc, cm = compute_metrics_binary(y_true, y_pred, y_prob)

    return total_loss, acc, prec, rec, f1, auroc, cm


def compute_metrics_binary(y_true, y_pred, y_prob):
    """
    y_true, y_pred: 1D Long tensors on CPU
    """
    TP = ((y_pred == 1) & (y_true == 1)).sum().item()
    TN = ((y_pred == 0) & (y_true == 0)).sum().item()
    FP = ((y_pred == 1) & (y_true == 0)).sum().item()
    FN = ((y_pred == 0) & (y_true == 1)).sum().item()

    acc = (TP + TN) / max(TP + TN + FP + FN, 1)
    prec = TP / max(TP + FP, 1)
    rec = TP / max(TP + FN, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    auroc = roc_auc_score(y_true.numpy(), y_prob.numpy())

    cm = torch.tensor([[TN, FP],
                       [FN, TP]])

    return acc, prec, rec, f1, auroc, cm

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU
        torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
        torch.backends.cudnn.benchmark = False  # Disable to prevent variability

def compute_truncation_stats(dataset, tokenizer, max_len):
    texts = [text for text, _ in dataset.samples]

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        return_length=True,
        padding=False,   # important
    )

    lengths = torch.tensor(tokenized["length"])

    stats = {
        "mean_tokens": lengths.float().mean().item(),
        "max_tokens": lengths.max().item(),
        "pct_truncated": (lengths == max_len).float().mean().item() * 100,
    }
    return stats

def subsample_per_class(
        samples: list[TextSample],
        max_per_class: int | None,
        seed: int = 42,
):
    if max_per_class is None:
        return samples

    g = torch.Generator().manual_seed(seed)

    by_class = {}
    for text, label in samples:
        by_class.setdefault(label, []).append((text, label))

    out = []
    for label, items in by_class.items():
        n = min(len(items), max_per_class)
        idx = torch.randperm(len(items), generator=g)[:n]
        out.extend([items[i] for i in idx])

    return out

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
        ("Val AUROC", "auroc"),
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

# === plot functions =============================================== End

# load data functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> begin
def load_mednli_as_text(config: dict) -> tuple[list[TextSample], list[TextSample]]:
    train_jsonl = config["train_jsonl"]
    test_jsonl  = config["test_jsonl"]
    sep         = config.get("sep", " [SEP] ")
    balance     = config.get("balance", False)

    train = load_all_data([train_jsonl])
    test  = load_all_data([test_jsonl])

    if balance:
        train = balance_data(train)
        test  = balance_data(test)

    def to_text(ex):
        return ex["premise"] + sep + ex["hypothesis"]

    train_samples = [(to_text(ex), ex["label"]) for ex in train]
    test_samples  = [(to_text(ex), ex["label"]) for ex in test]

    return train_samples, test_samples


def load_aclImdb_as_text(config: dict) -> tuple[list[TextSample], list[TextSample]]:
    train_dir = config["train_dir"]
    test_dir  = config["test_dir"]
    encoding  = config.get("encoding", "utf-8")

    train_per_class = config.get("train_per_class")   # NEW
    test_per_class  = config.get("test_per_class")    # NEW
    seed             = config.get("seed", 42)

    def load_split(split_dir):
        samples = []
        for label_name, label in [("neg", 0), ("pos", 1)]:
            class_dir = os.path.join(split_dir, label_name)
            for fname in os.listdir(class_dir):
                if fname.endswith(".txt"):
                    with open(os.path.join(class_dir, fname), encoding=encoding) as f:
                        samples.append((f.read().strip(), label))
        return samples

    train_samples = load_split(train_dir)
    test_samples  = load_split(test_dir)

    train_samples = subsample_per_class(
        train_samples,
        max_per_class=train_per_class,
        seed=seed,
    )
    test_samples = subsample_per_class(
        test_samples,
        max_per_class=test_per_class,
        seed=seed + 1,   # avoid overlap bias
    )

    return train_samples, test_samples

def load_jsonl_text_mimic4_discharge(
        config: dict
) -> tuple[list[TextSample], list[TextSample]]:
    """
    Load train/test JSONL files with structure:
      {"text": "...", "label": 0, "id": "..."}

    Returns:
      train_samples, test_samples
      where each sample is (text: str, label: int)
    """

    train_jsonl = config["train_jsonl"]
    test_jsonl  = config["test_jsonl"]
    balance     = config.get("balance", False)

    train = load_all_data([train_jsonl])
    test  = load_all_data([test_jsonl])

    if balance:
        train = balance_data(train)
        test  = balance_data(test)

    def to_sample(ex):
        # defensive casting
        text  = str(ex["text"])
        label = int(ex["label"])
        return text, label

    train_samples = [to_sample(ex) for ex in train]
    test_samples  = [to_sample(ex) for ex in test]

    return train_samples, test_samples

def preshuffle_samples(samples, seed=42):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(samples), generator=g)
    return [samples[i] for i in perm]

# load data functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> end

# Stats and Logger functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Begin
from collections import defaultdict

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

def temporal_smoothness(h):
    return (h[:, 1:] - h[:, :-1]).norm(dim=-1).mean()

def spectral_tail(h, k=20):
    H = h.reshape(-1, h.size(-1))
    s = torch.linalg.svdvals(H)
    return s[:k].mean()

def log_gradients(model, stats):
    total = 0.0
    count = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            g = p.grad.norm()
            stats.log(f"grad/{name}", g)
            total += g.item()
            count += 1

    # 🔑 THIS WAS MISSING
    stats.log("grad_norm", total / max(count, 1))

def read_max_len(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content_string = file.read()
        return int(file_content_string)

def get_class_weights(use_class_weighting, train_ds, device):
    if use_class_weighting:
        n0 = sum(1 for _, y in train_ds.samples if y == 0)
        n1 = sum(1 for _, y in train_ds.samples if y == 1)

        w1 = np.sqrt(n0 / max(n1, 1))          # n0 / max(n1, 1) or np.sqrt(n0 / max(n1, 1))
        class_weights = torch.tensor(
            [1.0, w1],
            device=device,
            dtype=torch.float32
        )
    else:
        class_weights = None

    if use_class_weighting:
        print(f"class_weights = {class_weights.tolist()}")

    return class_weights

def print_metrics(acc_p, prec_p, rec_p, f1_p, auroc_p, loss_plain,   val_loss_p,
                  acc_r, prec_r, rec_r, f1_r, auroc_r, loss_proj,    val_loss_r):

    header = f"{'Metric':<10} | {'PLAIN':>10} | {'PROJ':>10} |  Formula"
    print(header)
    print("-" * len(header))

    def fmt(x):
        return f"{x:.4f}"

    rows = [
        ("Accuracy",   acc_p,        acc_r,        "(TP + TN) / (TP + TN + FP + FN)"),
        ("Precision",  prec_p,       prec_r,       "TP / (TP + FP)"),
        ("Recall",     rec_p,        rec_r,        "TP / (TP + FN)"),
        ("F1-score",   f1_p,         f1_r,         "2·TP / (2·TP + FP + FN)"),
        ("AUROC", auroc_p, auroc_r, "ROC AUC"),
        (None, None, None, None),   # spacer line
        ("Train loss", loss_plain,   loss_proj,    ""),
        ("Val loss",   val_loss_p,   val_loss_r,   ""),
    ]
    for name, v_plain, v_proj, formula in rows:
        if name is None:
            print("-" * len(header))
            continue
        print(
            f"{name:<10} | "
            f"{fmt(v_plain):>10} | "
            f"{fmt(v_proj):>10} | "
            f"  {formula}"
        )
# def folder_now():
#     current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#     return current_time
def name_now():
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return current_time

class MyLogger:
    def __init__(self, *files):
        self.files = files
    def write(self, msg):
        for f in self.files:
            f.write(msg)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()
    def isatty(self):
        return hasattr(self.files[0], "isatty") and self.files[0].isatty()

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

# Stats and Logger functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> End

seed_everything()

base_dir = os.path.dirname(__file__)
train_data_file = test_data_file = None
_before = _before()
# ========================================================================
# dataset_name        = "mednli"
# dataset_name        = "aclImdb"
# dataset_name        = "snli"
dataset_name        = "qqp"
# dataset_name        = "mimic4_discharge"
#
# encoder_model_name    = "bert-base-uncased"
encoder_model_name  = "allenai/longformer-base-4096"
#
# in case "mimic4_discharge it will be read from file
# max_len             = 128 # 4096 # 128 # 512 # (for bert max = 512, for allenai/longformer-base-4096 max = 4096) 2048

batch_size          = 32 # 16 # 64 32 6 256
# Dc_list         = [16, 64, 128]
# Dc_list         = [8, 32, 64]
# Dc_list         = [64, 128, 256, 512, 1024]
# Dc_list         = [64, 128]
Dc_list             = [64]
lambda_entropy      = 1e-3
preshuffle_seed     = 42 # 61 # 56 # 53 # 23 # 26 # 25 # 30 # 35 # 40 # 42
#
projector_type      = "both" # None | "feature" | "sequence" | "both"
projector_location  = "hidden"   # hidden / input
#
epochs          = 30 # 30 # 10 20
#
lr1             = 1.e-2
lr2             = 1.e-5
# lr1             = 5.e-3
# lr2             = 1.e-4

lr              = [lr1, lr2]
lr_change       = 'linear' # linear  step
lr_step_epoch   = 3
#
use_alpha_h_optimizer       = False
alpha_h_lr                  = 1.e-4 # 1.e-4 # learning rate for alpha_h
#
use_MLP                     = True
#
num_refinement_steps_plain  = 1
num_refinement_steps_proj   = 1
project_only_once           = True   # project only one time at the end of refinement steps
normalize_proj              = True
#
num_smoothing_steps         = 1      # iterative Projector action
if projector_type is None: num_smoothing_steps = 0
#
use_class_weighting         = False
use_eta_schedule            = False
eta_constant                = None # 0.5 # None # if not None overrides other eta options
#
# with_noise                  = True
#
# pdf_file_name = f"{dataset_name}_training_metrics.pdf"

name_right_now = name_now()

data_dir = f"{base_dir}/data/{dataset_name}"
case = f'output_{name_right_now}'
out_dir = f'{data_dir}/{case}'  # individual in every run
images_dir = f"{out_dir}/images"
log_file_name  = f'{name_right_now}_printout.log'
log_file = f'{data_dir}/logs/{log_file_name}'
os.makedirs(images_dir, exist_ok=True)

if dataset_name == "mednli" or dataset_name == "snli":
    max_len     = 128
    with_noise  = 'N/A'
    dataset_config = {
        "train_jsonl": f"{data_dir}/output/train.jsonl",
        "test_jsonl":  f"{data_dir}/output/test.jsonl",
        "balance": True,
        "seed": 42,
    }
elif dataset_name == "qqp":
    max_len     = 128
    with_noise  = True    # False True
    train_file  = f"{data_dir}/output/train.jsonl"
    test_file   = f"{data_dir}/output/test.jsonl"
    if with_noise:
        train_file  = train_file.replace('.', '_noise.')
        test_file   = test_file.replace('.', '_noise.')
    dataset_config = {
        "train_jsonl": train_file,
        "test_jsonl":  test_file,
        "balance": True,  # it is balanced or has a given fraction by input
        "seed": 42,
    }
elif dataset_name == "aclImdb":
    with_noise = 'N/A'
    dataset_config = {
        "train_dir": f"{data_dir}/input/train",
        "test_dir":  f"{data_dir}/input/test",

        "train_per_class": 10000,   # ← e.g. 5k pos + 5k neg
        "test_per_class":  2000,   # ← e.g. 1k pos + 1k neg
        "seed": 42,
    }
elif dataset_name == "mimic4_discharge":
    with_noise = 'N/A'
    max_len = read_max_len(f"{data_dir}/output/max_tokens.txt")
    balanced = True
    balanced_str = '_balanced' if balanced else ''
    dataset_config = {
        "train_jsonl": f"{data_dir}/output/train{balanced_str}.jsonl",
        "test_jsonl":  f"{data_dir}/output/test{balanced_str}.jsonl",
        "seed": 42,
    }
else:
    raise ValueError(f"Unknown dataset: {dataset_name}")

T = max_len
Tc  = T // 4

print(f'\nmax_len: {max_len}, T: {T}, Tc: {Tc}')

DATASET_REGISTRY = {
    "mednli":           load_mednli_as_text,
    "snli":             load_mednli_as_text,
    "aclImdb":          load_aclImdb_as_text,
    "qqp":              load_mednli_as_text,
    "mimic4_discharge": load_jsonl_text_mimic4_discharge,
}
# ========================================================================

new_vars = _after(_before)
# Activate printout logging
if log_file is not None:
    log_file = open(log_file, "a")
    sys.stdout = MyLogger(sys.stdout, log_file)
# print input parameters
for var in new_vars:
    print(var)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'device: {device}')

encoder = TextEncoder(
    model_name=encoder_model_name,
    max_len=max_len
).to(device)

D = encoder.encoder.config.hidden_size  # 768

seed_everything()
model_plain = TinyTransformer(D, Dc_list=None, T=T, Tc=Tc, projector_type=None,
                              use_MLP=use_MLP,
                              num_refinement_steps=num_refinement_steps_plain)

seed_everything()
model_proj  = TinyTransformer(D, Dc_list=Dc_list, T=T, Tc=Tc, projector_type=projector_type,
                              use_MLP=use_MLP,
                              num_smoothing_steps=num_smoothing_steps,
                              num_refinement_steps=num_refinement_steps_proj,
                              proj_once=project_only_once, normalize_proj=normalize_proj,
                              projector_location=projector_location,
                              eta_constant=eta_constant)

if projector_type is not None:
    assert model_proj.projs is not None
    assert hasattr(model_proj, "projs") and len(model_proj.projs) > 0

train_samples, test_samples = DATASET_REGISTRY[dataset_name](dataset_config)

train_samples = preshuffle_samples(train_samples, seed=preshuffle_seed)

train_ds = TextBinaryDataset(train_samples)
test_ds  = TextBinaryDataset(test_samples)

class_weights = get_class_weights(use_class_weighting, train_ds, device)

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=False, # train samples were pre-shuffled
    collate_fn=collate_text
)
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_text,
)
print_data_diagnostics(
    train_or_test="TRAIN",
    dataset=train_ds,
    dataloader=train_loader,
    encoder=encoder,
    D=D,
    Dc_list=Dc_list,
    device=device,
)
print_data_diagnostics(
    train_or_test="TEST",
    dataset=test_ds,
    dataloader=test_loader,
    encoder=encoder,
    D=D,
    Dc_list=Dc_list,
    device=device,
)

balance_train_dict  = print_label_balance("TRAIN", train_ds)
balance_test_dict   = print_label_balance("TEST", test_ds)

train_stats = compute_truncation_stats(
    train_ds,
    encoder.tokenizer,
    encoder.max_len,
)
test_stats = compute_truncation_stats(
    test_ds,
    encoder.tokenizer,
    encoder.max_len,
)
print("\n=== Corpus token stats ===")
print(f"TRAIN: mean={train_stats['mean_tokens']:.1f}, "
      f"max={train_stats['max_tokens']}, "
      f"truncated={train_stats['pct_truncated']:.2f}%")

print(f"TEST : mean={test_stats['mean_tokens']:.1f}, "
      f"max={test_stats['max_tokens']}, "
      f"truncated={test_stats['pct_truncated']:.2f}%")

#---------------------------
# lambda_entropy = 1e-3
#---------------------------

history =  train_dual(
    model_plain,
    model_proj,
    encoder,
    train_loader,
    test_loader,
    epochs=epochs,
    lr=lr,
    lr_change=lr_change,
    lr_step_epoch=lr_step_epoch,
    device=device,
    use_alpha_h_optimizer=use_alpha_h_optimizer,
    alpha_h_lr=alpha_h_lr,
    lambda_entropy=lambda_entropy,
    class_weights=class_weights,
    images_dir=images_dir,
    use_eta_schedule=use_eta_schedule,
)
print("\n================ Input Summary ================\n")
print(
    f"date                                = {'{:%Y-%m-%d %H:%M:%S}'.format(datetime.now())}\n"
    f'dataset                             = {dataset_name}\n'
    f'balance_train_dict                  = {balance_train_dict}\n'
    f'balance_test_dict                   = {balance_test_dict}\n'
    f'len(train_samples)                  = {len(train_ds)}\n'
    f'len(test_samples)                   = {len(test_ds)}\n'
    f'max_len                             = {max_len}\n'
    f'batch_size                          = {batch_size}\n'
    f'D                                   = {D}\n'
    f'Dc_list                             = {Dc_list}\n'
    f'Tc                                  = {Tc}\n'
    f'epochs                              = {epochs}\n'
    f'lr                                  = {lr}\n'
    f'lr_change                           = {lr_change}\n'
    f'lr_step_epoch                       = {lr_step_epoch}\n'
    f'use_alpha_h_optimizer               = {use_alpha_h_optimizer}\n'
    f'projector_type                      = {projector_type}\n'
    f'model_proj.use_MLP                  = {model_proj.use_MLP}\n'
    f'model_plain.use_MLP                 = {model_plain.use_MLP}\n'
    f'model_proj.num_smoothing_steps      = {model_proj.num_smoothing_steps}\n'
    f'model_proj.num_refinement_steps     = {model_proj.num_refinement_steps}\n'
    f'model_plain.num_refinement_steps    = {model_plain.num_refinement_steps}\n'
    f'model_proj.proj_once                = {model_proj.proj_once}\n'
    f'model_proj.normalize_proj           = {model_proj.normalize_proj}\n'
    f'use_class_weighting                 = {use_class_weighting}\n'
    f'with_noise                          = {with_noise}\n'
)

if model_proj.projector_type and hasattr(model_proj.projs[0], "A_shape"):
    print()
    print("\n[Projector] Linear system summary:")
    a_shape = model_proj.projs[0].A_shape
    print(f"  A   shape: {a_shape}, size of Dc")
    rhs_shape = model_proj.projs[0].RHS_shape
    print(f"  RHS shape: {rhs_shape}, {rhs_shape[1]} is B × T (batch size × sequence length)")

