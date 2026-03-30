"""
Architecture and Training Pipeline Figure Generator
Produces a publication-quality figure for the paper.

Two outputs:
  1. figures/model_architecture.pdf  — full pipeline diagram
  2. figures/model_architecture.png  — high-DPI raster copy
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    'bg'        : '#FAFAFA',
    'data'      : '#EAF4FB',   # light blue  — data / input boxes
    'data_bd'   : '#2E86AB',   # blue border
    'embed'     : '#FFF3CD',   # amber       — embedding layer (frozen)
    'embed_bd'  : '#E67E22',
    'lstm'      : '#E8F5E9',   # green       — LSTM layers
    'lstm_bd'   : '#27AE60',
    'attn'      : '#F3E5F5',   # purple      — attention
    'attn_bd'   : '#8E44AD',
    'pool'      : '#FCE4EC',   # pink        — pooling / fc
    'pool_bd'   : '#C0392B',
    'out'       : '#FFF9C4',   # yellow      — output
    'out_bd'    : '#F39C12',
    'ddp'       : '#E8EAF6',   # indigo      — DDP wrapper
    'ddp_bd'    : '#3949AB',
    'dedup'     : '#E0F2F1',   # teal        — deduplication
    'dedup_bd'  : '#00897B',
    'arrow'     : '#34495E',
    'frozen_lbl': '#C0392B',
    'text_dark' : '#1A1A2E',
    'text_mid'  : '#34495E',
    'text_light': '#7F8C8D',
    'grid'      : '#E0E0E0',
}

FONT = 'DejaVu Sans'

def rbox(ax, x, y, w, h, fc, ec, lw=1.5, radius=0.012, zorder=3, alpha=1.0):
    """Draw a rounded rectangle patch."""
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc,
        zorder=zorder, alpha=alpha
    )
    ax.add_patch(box)
    return box

def arrow(ax, x0, y0, x1, y1, color='#34495E', lw=1.6,
          style='->', size=12, zorder=4, connectionstyle='arc3,rad=0.0'):
    ax.annotate(
        '', xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=f'->', color=color,
            lw=lw, mutation_scale=size,
            connectionstyle=connectionstyle
        ),
        zorder=zorder
    )

def label(ax, x, y, txt, fontsize=8, color='#1A1A2E', bold=False,
          ha='center', va='center', zorder=5):
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, txt, fontsize=fontsize, color=color, fontweight=weight,
            ha=ha, va='center', zorder=zorder, fontfamily=FONT)

def sublabel(ax, x, y, txt, fontsize=6.5, color='#7F8C8D',
             ha='center', zorder=5):
    ax.text(x, y, txt, fontsize=fontsize, color=color,
            ha=ha, va='center', zorder=zorder, fontfamily=FONT, style='italic')

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11), facecolor=C['bg'])

# ── Two sub-axes: left = data pipeline, right = model forward pass ─────────
ax_main = fig.add_axes([0.0, 0.0, 1.0, 1.0], facecolor=C['bg'])
ax_main.set_xlim(0, 1)
ax_main.set_ylim(0, 1)
ax_main.axis('off')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A — Data & Deduplication Pipeline  (left third)
# ─────────────────────────────────────────────────────────────────────────────
ax_main.text(0.12, 0.965, 'A  ·  Data & Deduplication Pipeline',
             fontsize=10, fontweight='bold', color=C['text_dark'],
             ha='center', va='center', fontfamily=FONT)

# Background panel
rbox(ax_main, 0.12, 0.50, 0.215, 0.90, '#F0F4F8', '#BDBDBD', lw=1.0, radius=0.02, zorder=1, alpha=0.5)

# ── Raw Datasets ──
rbox(ax_main, 0.12, 0.890, 0.185, 0.055, C['data'], C['data_bd'], lw=1.8, radius=0.012)
label(ax_main, 0.12, 0.896, 'Raw Source Code Datasets', 8, bold=True)
sublabel(ax_main, 0.12, 0.882, 'Juliet C (NSA SARD)  ·  Devign  ·  BugsInPy')

# ── Tree-sitter parsing ──
rbox(ax_main, 0.12, 0.800, 0.185, 0.055, C['dedup'], C['dedup_bd'], lw=1.5, radius=0.012)
label(ax_main, 0.12, 0.806, 'Tree-sitter Function Extraction', 8, bold=True)
sublabel(ax_main, 0.12, 0.792, 'Remove comments & whitespace  |  min 32 / max 4096 tokens')

arrow(ax_main, 0.12, 0.862, 0.12, 0.828)

# ── SimHash deduplication ──
rbox(ax_main, 0.12, 0.703, 0.185, 0.063, C['dedup'], C['dedup_bd'], lw=1.8, radius=0.012)
label(ax_main, 0.12, 0.715, 'SimHash Deduplication', 8.5, bold=True)
sublabel(ax_main, 0.12, 0.702, 'k-mer shingling  →  64-bit fingerprint')
sublabel(ax_main, 0.12, 0.691, 'Hamming distance threshold  k ∈ {1, 2, …, 12}')

arrow(ax_main, 0.12, 0.772, 0.12, 0.735)

# k annotation
ax_main.annotate('', xy=(0.215, 0.703), xytext=(0.185, 0.703),
    arrowprops=dict(arrowstyle='->', color=C['dedup_bd'], lw=1.4))
ax_main.text(0.218, 0.703, '12 deduplicated\ndataset variants\nper corpus',
    fontsize=6.5, color=C['dedup_bd'], va='center', fontfamily=FONT)

# ── Tokenization ──
rbox(ax_main, 0.12, 0.614, 0.185, 0.055, C['data'], C['data_bd'], lw=1.5, radius=0.012)
label(ax_main, 0.12, 0.620, 'Tokenization  (Qwen 2.5-7B)', 8, bold=True)
sublabel(ax_main, 0.12, 0.606, 'Vocab: 49,152  |  Pad/truncate → 4,096 tokens')

arrow(ax_main, 0.12, 0.671, 0.12, 0.642)

# ── Balancing ──
rbox(ax_main, 0.12, 0.527, 0.185, 0.055, C['data'], C['data_bd'], lw=1.5, radius=0.012)
label(ax_main, 0.12, 0.533, 'Class & CWE Balancing', 8, bold=True)
sublabel(ax_main, 0.12, 0.519, 'CVE-stratified sampling  |  50/50 vuln/secure')

arrow(ax_main, 0.12, 0.582, 0.12, 0.555)

# ── Train / Val / Test split ──
rbox(ax_main, 0.12, 0.438, 0.185, 0.055, C['data'], C['data_bd'], lw=1.5, radius=0.012)
label(ax_main, 0.12, 0.444, 'Train / Val / Test Split', 8, bold=True)
sublabel(ax_main, 0.12, 0.430, '80% train  ·  10% val  ·  10% test')

arrow(ax_main, 0.12, 0.499, 0.12, 0.466)

# ── PyTorch Tensors ──
rbox(ax_main, 0.12, 0.350, 0.185, 0.055, C['data'], C['data_bd'], lw=1.8, radius=0.012)
label(ax_main, 0.12, 0.356, 'PyTorch .pt Tensors', 8, bold=True)
sublabel(ax_main, 0.12, 0.342, '{train,val,test}_{sequences,labels,cwe_indices}.pt')

arrow(ax_main, 0.12, 0.410, 0.12, 0.378)

# ── OOD dataset note ──
rbox(ax_main, 0.12, 0.248, 0.185, 0.065, '#FFF8E1', '#F9A825', lw=1.4, radius=0.012)
label(ax_main, 0.12, 0.261, 'OOD Evaluation Set', 8, bold=True, color='#E65100')
sublabel(ax_main, 0.12, 0.248, 'Exp 1–3: Juliet C → Devign (real-world C)')
sublabel(ax_main, 0.12, 0.236, 'Exp 4:   Devign → Juliet C (synthetic)')

arrow(ax_main, 0.12, 0.322, 0.12, 0.281)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION B — Model Architecture (centre)
# ─────────────────────────────────────────────────────────────────────────────
ax_main.text(0.52, 0.965, 'B  ·  Model Architecture  (forward pass — single sample)',
             fontsize=10, fontweight='bold', color=C['text_dark'],
             ha='center', va='center', fontfamily=FONT)

cx = 0.52   # centre x of model column
rbox(ax_main, cx, 0.50, 0.395, 0.90, '#F0F4F8', '#BDBDBD', lw=1.0, radius=0.02, zorder=1, alpha=0.5)

# ── Input token sequence ──
rbox(ax_main, cx, 0.890, 0.340, 0.048, C['data'], C['data_bd'], lw=1.8, radius=0.010)
label(ax_main, cx, 0.898, 'Token Sequence   x ∈ ℤ^{4096}', 8.5, bold=True)
sublabel(ax_main, cx, 0.882, 'Padded integer token IDs  (max length 4,096)')

# ── Frozen Embedding ──
rbox(ax_main, cx, 0.800, 0.340, 0.065, C['embed'], C['embed_bd'], lw=2.0, radius=0.010)
label(ax_main, cx, 0.815, 'Frozen Embedding Layer  E', 8.5, bold=True, color='#7D3C00')
sublabel(ax_main, cx, 0.802, 'E ∈ ℝ^{V × d}  |  frozen (no gradient)')
sublabel(ax_main, cx, 0.789, 'aiXcoder-7B: V=49,152  d=4,096')

# Frozen badge
rbox(ax_main, 0.685, 0.802, 0.055, 0.022, '#FFD7D7', C['frozen_lbl'], lw=1.2, radius=0.006)
ax_main.text(0.685, 0.802, '[FROZEN]', fontsize=6.5, color=C['frozen_lbl'],
             ha='center', va='center', fontweight='bold', fontfamily=FONT)

arrow(ax_main, cx, 0.864, cx, 0.833)
ax_main.text(cx+0.005, 0.849, 'lookup  →  drop(E[x])', fontsize=6.5,
             color=C['text_light'], ha='left', va='center', fontfamily=FONT)

# drop annotation
ax_main.text(cx, 0.776, 'embedded  ∈  ℝ^{B × 4096 × 4096}   (dropout p=0.5)',
             fontsize=6.5, color=C['text_light'], ha='center', va='center', fontfamily=FONT)

arrow(ax_main, cx, 0.767, cx, 0.748)

# ── Bi-LSTM ──
rbox(ax_main, cx, 0.685, 0.340, 0.110, C['lstm'], C['lstm_bd'], lw=2.0, radius=0.010)
label(ax_main, cx, 0.735, '2-Layer Bidirectional LSTM', 8.5, bold=True, color='#1B5E20')
sublabel(ax_main, cx, 0.722, 'hidden dim h = 256  per direction')

# forward / backward annotation
for sign, lbl in [(+1, '→ Forward'), (-1, '← Backward')]:
    rx = cx + sign * 0.08
    rbox(ax_main, rx, 0.695, 0.09, 0.030, '#D5F5E3', C['lstm_bd'], lw=1.1, radius=0.007)
    ax_main.text(rx, 0.695, lbl, fontsize=6.5, color='#1B5E20',
                 ha='center', va='center', fontfamily=FONT)
sublabel(ax_main, cx, 0.680, 'Layer 1  +  Layer 2  (inter-layer dropout p=0.5)')
sublabel(ax_main, cx, 0.667, 'output  h_t ∈  ℝ^{B × 4096 × 512}   (256 × 2 dirs)')

arrow(ax_main, cx, 0.639, cx, 0.620)
ax_main.text(cx+0.005, 0.630, 'h_t  ∈  ℝ^{B × T × 512}', fontsize=6.5,
             color=C['text_light'], ha='left', va='center', fontfamily=FONT)

# ── Multi-Head Attention ──
rbox(ax_main, cx, 0.568, 0.340, 0.090, C['attn'], C['attn_bd'], lw=2.0, radius=0.010)
label(ax_main, cx, 0.610, '8-Head Self-Attention', 8.5, bold=True, color='#4A148C')
sublabel(ax_main, cx, 0.597, 'Q = K = V = h_t   (self-attention)')
sublabel(ax_main, cx, 0.584, 'd_k = 512 / 8 = 64  per head')

# Q K V mini boxes
for i, lbl in enumerate(['Q', 'K', 'V']):
    bx = cx + (i-1) * 0.048
    rbox(ax_main, bx, 0.568, 0.030, 0.022, '#EDE7F6', C['attn_bd'], lw=1.0, radius=0.005)
    ax_main.text(bx, 0.568, lbl, fontsize=7, color=C['attn_bd'],
                 ha='center', va='center', fontweight='bold', fontfamily=FONT)
sublabel(ax_main, cx, 0.553, 'Dropout (p=0.5) on attention weights')

arrow(ax_main, cx, 0.522, cx, 0.503)

# ── Residual + LayerNorm ──
rbox(ax_main, cx, 0.482, 0.340, 0.038, '#EAF7FB', '#2980B9', lw=1.6, radius=0.008)
label(ax_main, cx, 0.489, 'Residual Connection  +  Layer Normalisation', 8, bold=True, color='#1A5276')
sublabel(ax_main, cx, 0.475, 'LayerNorm( h_t + attn_output )   ∈  ℝ^{B × T × 512}')

arrow(ax_main, cx, 0.463, cx, 0.443)

# ── Global Average Pooling ──
rbox(ax_main, cx, 0.422, 0.340, 0.038, C['pool'], C['pool_bd'], lw=1.8, radius=0.008)
label(ax_main, cx, 0.429, 'Global Average Pooling   (dim=T)', 8, bold=True, color='#7B241C')
sublabel(ax_main, cx, 0.415, 'mean over sequence  →  ℝ^{B × 512}')

arrow(ax_main, cx, 0.403, cx, 0.382)
ax_main.text(cx+0.005, 0.393, 'drop( · )  p=0.5', fontsize=6.5,
             color=C['text_light'], ha='left', va='center', fontfamily=FONT)

# ── Linear + Sigmoid ──
rbox(ax_main, cx, 0.358, 0.340, 0.048, C['out'], C['out_bd'], lw=1.8, radius=0.010)
label(ax_main, cx, 0.370, 'Linear (512 → 1)  +  Sigmoid', 8.5, bold=True, color='#7D6608')
sublabel(ax_main, cx, 0.356, 'ŷ = σ(W · pooled + b)   ∈  (0, 1)')

arrow(ax_main, cx, 0.334, cx, 0.314)

# ── Output ──
rbox(ax_main, cx, 0.292, 0.340, 0.040, C['out'], C['out_bd'], lw=2.0, radius=0.010)
label(ax_main, cx, 0.300, 'P( vulnerable | x )', 9, bold=True, color='#784212')
sublabel(ax_main, cx, 0.285, 'BCE Loss  ·  threshold 0.5  →  {secure, vulnerable}')

# ── Loss + Backprop annotation ──
rbox(ax_main, cx, 0.218, 0.340, 0.055, '#F9EBEA', C['pool_bd'], lw=1.4, radius=0.010, alpha=0.7)
label(ax_main, cx, 0.233, 'Training Objective', 8, bold=True, color=C['pool_bd'])
sublabel(ax_main, cx, 0.220, 'ℒ = BCE(ŷ, y)   ·   Adam (lr=0.001)')
sublabel(ax_main, cx, 0.207, 'Gradient clipping ‖∇‖ ≤ 1.0  ·  Early stopping (patience=5)')

ax_main.annotate(
    '', xy=(0.698, 0.292), xytext=(0.698, 0.218),
    arrowprops=dict(arrowstyle='<-', color=C['pool_bd'], lw=1.4,
                    connectionstyle='arc3,rad=0.0')
)
ax_main.text(0.703, 0.255, '∂ℒ/∂θ\n(backprop)', fontsize=6, color=C['pool_bd'],
             ha='left', va='center', fontfamily=FONT)

# Frozen — no grad arrow label
ax_main.annotate(
    '', xy=(0.698, 0.800), xytext=(0.698, 0.760),
    arrowprops=dict(arrowstyle='-|>', color=C['frozen_lbl'], lw=1.2,
                    connectionstyle='arc3,rad=0.0')
)
ax_main.text(0.703, 0.780, 'embedding\nfrozen\n(no grad)', fontsize=6,
             color=C['frozen_lbl'], ha='left', va='center', fontfamily=FONT)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION C — DDP Training & Evaluation Protocol (right column)
# ─────────────────────────────────────────────────────────────────────────────
ax_main.text(0.875, 0.965, 'C  ·  Training & Evaluation Protocol',
             fontsize=10, fontweight='bold', color=C['text_dark'],
             ha='center', va='center', fontfamily=FONT)

rx = 0.875
rbox(ax_main, rx, 0.50, 0.215, 0.90, '#F0F4F8', '#BDBDBD', lw=1.0, radius=0.02, zorder=1, alpha=0.5)

# ── DDP wrapper ──
rbox(ax_main, rx, 0.885, 0.190, 0.065, C['ddp'], C['ddp_bd'], lw=2.0, radius=0.012)
label(ax_main, rx, 0.900, 'DistributedDataParallel (DDP)', 8.5, bold=True, color='#1A237E')
sublabel(ax_main, rx, 0.887, '8 × GPU  ·  NCCL backend')
sublabel(ax_main, rx, 0.874, 'Per-GPU batch 16  →  effective batch 128')

# 8 GPU boxes
for i in range(8):
    gx = rx - 0.082 + i * 0.024
    rbox(ax_main, gx, 0.856, 0.018, 0.016, '#C5CAE9', C['ddp_bd'], lw=0.8, radius=0.004)
    ax_main.text(gx, 0.856, f'G{i}', fontsize=5, color='#1A237E',
                 ha='center', va='center', fontfamily=FONT)

arrow(ax_main, rx, 0.843, rx, 0.820)

# ── Multi-seed training ──
rbox(ax_main, rx, 0.793, 0.190, 0.050, C['ddp'], C['ddp_bd'], lw=1.6, radius=0.010)
label(ax_main, rx, 0.803, '20 Independent Seeds', 8.5, bold=True, color='#1A237E')
sublabel(ax_main, rx, 0.790, 'seeds 1 – 20  for statistical robustness')

arrow(ax_main, rx, 0.768, rx, 0.748)

# ── k sweep ──
rbox(ax_main, rx, 0.720, 0.190, 0.050, C['dedup'], C['dedup_bd'], lw=1.6, radius=0.010)
label(ax_main, rx, 0.730, 'k-Value Sweep', 8.5, bold=True, color='#00695C')
sublabel(ax_main, rx, 0.717, 'k ∈ {1, 2, …, 12}  ×  20 seeds  =  240 runs')

arrow(ax_main, rx, 0.695, rx, 0.675)

# ── Epoch loop ──
rbox(ax_main, rx, 0.645, 0.190, 0.055, C['lstm'], C['lstm_bd'], lw=1.6, radius=0.010)
label(ax_main, rx, 0.660, 'Training Loop', 8.5, bold=True, color='#1B5E20')
sublabel(ax_main, rx, 0.647, 'Max 50 epochs  ·  patience = 5')
sublabel(ax_main, rx, 0.634, 'Best val-loss checkpoint saved')

arrow(ax_main, rx, 0.617, rx, 0.597)

# ── Validation ──
rbox(ax_main, rx, 0.568, 0.190, 0.050, C['lstm'], C['lstm_bd'], lw=1.4, radius=0.010)
label(ax_main, rx, 0.578, 'Validation (per epoch)', 8, bold=True, color='#1B5E20')
sublabel(ax_main, rx, 0.565, 'Val AUROC  ·  Val Loss  →  early stop')

arrow(ax_main, rx, 0.543, rx, 0.523)

# ── Test evaluation ──
rbox(ax_main, rx, 0.495, 0.190, 0.050, C['pool'], C['pool_bd'], lw=1.6, radius=0.010)
label(ax_main, rx, 0.505, 'In-Distribution Evaluation', 8.5, bold=True, color='#7B241C')
sublabel(ax_main, rx, 0.492, 'Test split  →  AUROC, F1, Prec, Recall')

arrow(ax_main, rx, 0.470, rx, 0.450)

# ── OOD evaluation ──
rbox(ax_main, rx, 0.420, 0.190, 0.055, '#FFF3E0', '#E65100', lw=1.8, radius=0.010)
label(ax_main, rx, 0.435, 'OOD Evaluation', 8.5, bold=True, color='#BF360C')
sublabel(ax_main, rx, 0.422, 'Held-out cross-domain dataset')
sublabel(ax_main, rx, 0.409, 'Generalization gap Δ = AUROCtest − AUROCood')

arrow(ax_main, rx, 0.392, rx, 0.372)

# ── Statistics ──
rbox(ax_main, rx, 0.340, 0.190, 0.063, '#E0F7FA', '#006064', lw=1.6, radius=0.010)
label(ax_main, rx, 0.360, 'Statistical Analysis', 8.5, bold=True, color='#004D40')
sublabel(ax_main, rx, 0.347, '95% CI over 20 seeds')
sublabel(ax_main, rx, 0.334, 'Welch t-test  ·  one-way ANOVA')
sublabel(ax_main, rx, 0.321, "Cohen's d  effect size")

arrow(ax_main, rx, 0.308, rx, 0.288)

# ── Outputs ──
rbox(ax_main, rx, 0.258, 0.190, 0.055, C['out'], C['out_bd'], lw=1.8, radius=0.010)
label(ax_main, rx, 0.273, 'Outputs per Experiment', 8.5, bold=True, color='#7D6608')
sublabel(ax_main, rx, 0.260, 'JSON results  ·  CSV summary')
sublabel(ax_main, rx, 0.247, 'Training curves  ·  Per-CWE metrics')

# ── 4 experiments summary box ──
rbox(ax_main, rx, 0.160, 0.190, 0.075, '#F3F3F3', '#607D8B', lw=1.4, radius=0.010)
label(ax_main, rx, 0.188, 'Experiment Summary', 8, bold=True, color='#37474F')
for i, (exp, detail) in enumerate([
    ('Exp 1', 'aiXcoder-7B   Juliet C → Devign'),
    ('Exp 2', 'DeepSeek-6.7B  Juliet C → Devign'),
    ('Exp 3', 'CodeLlama-7B  Juliet C → Devign'),
    ('Exp 4', 'DeepSeek-6.7B  Devign  → Juliet C'),
]):
    ey = 0.172 - i * 0.013
    ax_main.text(rx - 0.085, ey, exp, fontsize=6, color='#1A237E',
                 fontweight='bold', ha='left', va='center', fontfamily=FONT)
    ax_main.text(rx - 0.055, ey, detail, fontsize=6, color='#37474F',
                 ha='left', va='center', fontfamily=FONT)

# ─────────────────────────────────────────────────────────────────────────────
# Connecting arrows between panels
# ─────────────────────────────────────────────────────────────────────────────
# Data pipeline → Model (tensors feed into model)
arrow(ax_main, 0.213, 0.350, 0.323, 0.890,
      color='#2E86AB', lw=1.8, connectionstyle='arc3,rad=-0.18')
ax_main.text(0.265, 0.650, 'tokenised\ntensors', fontsize=7, color='#2E86AB',
             ha='center', va='center', fontfamily=FONT, style='italic')

# Model → Training protocol
arrow(ax_main, 0.700, 0.570, 0.770, 0.645,
      color='#27AE60', lw=1.8, connectionstyle='arc3,rad=-0.15')
ax_main.text(0.740, 0.630, 'model\nforward', fontsize=7, color='#27AE60',
             ha='center', va='center', fontfamily=FONT, style='italic')

# ─────────────────────────────────────────────────────────────────────────────
# Section dividers
# ─────────────────────────────────────────────────────────────────────────────
for xd in [0.238, 0.724]:
    ax_main.axvline(xd, ymin=0.02, ymax=0.97, color='#CFD8DC', lw=1.0, linestyle='--', zorder=0)

# ─────────────────────────────────────────────────────────────────────────────
# Legend — embedding sources
# ─────────────────────────────────────────────────────────────────────────────
legend_y = 0.063
rbox(ax_main, 0.50, legend_y, 0.97, 0.075, '#F5F5F5', '#90A4AE', lw=1.0, radius=0.015, zorder=1)
ax_main.text(0.50, legend_y + 0.025, 'Embedding Weight Sources  (frozen)',
             fontsize=8.5, fontweight='bold', color=C['text_dark'],
             ha='center', va='center', fontfamily=FONT)

sources = [
    ('aiXcoder-7B\n(Exp 1)', '#E67E22',
     'V=49,152  d=4,096\nLoaded from .pt checkpoint\ntok_embeddings.weight'),
    ('DeepSeek-Coder-6.7B\n(Exp 2, 4)', '#2980B9',
     'V=32,256  d=4,096\nLoaded from HuggingFace\nmodel.embed_tokens.weight'),
    ('CodeLlama-7B\n(Exp 3)', '#8E44AD',
     'V=32,000  d=4,096\nLoaded from HuggingFace\nmodel.embed_tokens.weight'),
]
for i, (name, color, detail) in enumerate(sources):
    sx = 0.16 + i * 0.27
    rbox(ax_main, sx, legend_y - 0.008, 0.240, 0.038, '#FAFAFA', color, lw=1.6, radius=0.008)
    ax_main.text(sx - 0.100, legend_y - 0.008, name, fontsize=7.5, color=color,
                 fontweight='bold', ha='left', va='center', fontfamily=FONT)
    ax_main.text(sx + 0.010, legend_y - 0.008, detail, fontsize=6.2, color='#555',
                 ha='left', va='center', fontfamily=FONT)

# ─────────────────────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────────────────────
fig.text(0.50, 0.994,
         'Architecture & Training Pipeline: '
         'BiLSTM-Attention Vulnerability Classifier with Pretrained LLM Embeddings',
         fontsize=11.5, fontweight='bold', color=C['text_dark'],
         ha='center', va='top', fontfamily=FONT)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
import os
os.makedirs('figures', exist_ok=True)

fig.savefig('figures/model_architecture.pdf', dpi=300, bbox_inches='tight',
            facecolor=C['bg'])
fig.savefig('figures/model_architecture.png', dpi=300, bbox_inches='tight',
            facecolor=C['bg'])

print("Saved: figures/model_architecture.pdf")
print("Saved: figures/model_architecture.png")
