"""
Publication-Quality Normality Analysis Figures
===============================================
Investigates whether the function-level size distributions in the
Juliet C and Devign datasets are normally distributed.

Produces:
  1. Histograms with overlaid normal + best-fit curves
  2. Q-Q (quantile-quantile) plots against the normal distribution
  3. Log-transformed distributions + Q-Q plots
  4. Statistical test summary panel (Shapiro–Wilk, D'Agostino–Pearson,
     Anderson–Darling, Kolmogorov–Smirnov)
  5. CDF comparison plots (empirical vs theoretical normal)
  6. Skewness/kurtosis annotated box-plot panel

All outputs saved to figures/ as PDF (vector) and PNG (300 DPI).
"""

import sqlite3, os, re, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy import stats as sp_stats

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── Style (matches generate_dataset_figures.py) ──────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.color': '#CCCCCC',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

os.makedirs('figures', exist_ok=True)

JULIET_DB  = 'datasets/juliet_c.db'
DEVIGN_DB  = 'datasets/devign.db'
JULIET_CLR = '#E67E22'
DEVIGN_CLR = '#2980B9'
VULN_CLR   = '#C0392B'
SEC_CLR    = '#2E86AB'
NORMAL_CLR = '#2ECC71'

# ============================================================================
# Data loading
# ============================================================================
def load_funcs(db):
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT grp, vuln, code FROM funcs").fetchall()
    conn.close()
    return rows

def is_vuln(v):
    if v is None: return False
    v = str(v).strip()
    return v != '' and v != '0'

print("Loading databases …")
juliet = load_funcs(JULIET_DB)
devign = load_funcs(DEVIGN_DB)

# Core feature arrays
juliet_chars = np.array([len(r[2]) for r in juliet])
juliet_lines = np.array([r[2].count('\n') + 1 for r in juliet])
devign_chars = np.array([len(r[2]) for r in devign])
devign_lines = np.array([r[2].count('\n') + 1 for r in devign])

print(f"  Juliet C: {len(juliet):,}   Devign: {len(devign):,}")


# ============================================================================
# Helper: run a battery of normality tests
# ============================================================================
def normality_battery(data, label, max_shapiro=5000):
    """Return a dict of test results.  Shapiro–Wilk is capped at 5 000
    samples (random sub-sample) because it is O(n²)."""
    results = {}

    # 1. Shapiro–Wilk (sub-sample for large N)
    if len(data) > max_shapiro:
        rng = np.random.default_rng(42)
        sub = rng.choice(data, max_shapiro, replace=False)
    else:
        sub = data
    stat, p = sp_stats.shapiro(sub)
    results['Shapiro–Wilk'] = {'stat': stat, 'p': p, 'n_used': len(sub)}

    # 2. D'Agostino–Pearson
    if len(data) >= 20:
        stat, p = sp_stats.normaltest(data)
        results["D'Agostino–Pearson"] = {'stat': stat, 'p': p}
    else:
        results["D'Agostino–Pearson"] = {'stat': np.nan, 'p': np.nan}

    # 3. Anderson–Darling
    ad = sp_stats.anderson(data, dist='norm')
    # Report the 5 % significance level
    idx = list(ad.significance_level).index(5.0) if 5.0 in ad.significance_level else 2
    results['Anderson–Darling'] = {
        'stat': ad.statistic,
        'crit_5pct': ad.critical_values[idx],
        'reject': ad.statistic > ad.critical_values[idx],
    }

    # 4. Kolmogorov–Smirnov (against fitted normal)
    mu, sigma = data.mean(), data.std(ddof=1)
    stat, p = sp_stats.kstest(data, 'norm', args=(mu, sigma))
    results['KS'] = {'stat': stat, 'p': p}

    # Descriptive
    results['_desc'] = {
        'n': len(data),
        'mean': data.mean(),
        'std': data.std(ddof=1),
        'skew': sp_stats.skew(data),
        'kurtosis': sp_stats.kurtosis(data),  # excess kurtosis
        'median': np.median(data),
    }
    return results


# Run all batteries ----------------------------------------------------------
datasets_info = [
    ('Juliet C — Chars',  juliet_chars, JULIET_CLR),
    ('Juliet C — Lines',  juliet_lines, JULIET_CLR),
    ('Devign — Chars',    devign_chars, DEVIGN_CLR),
    ('Devign — Lines',    devign_lines, DEVIGN_CLR),
]

all_results = {}
for label, arr, _ in datasets_info:
    all_results[label] = normality_battery(arr, label)
    log_arr = np.log1p(arr)
    all_results[label + ' (log)'] = normality_battery(log_arr, label + ' (log)')


# ============================================================================
# FIGURE 1 — Histogram + Normal overlay  (2 × 2)
# ============================================================================
print("\nFigure 1: Histograms with fitted normal overlay …")

fig1, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, (label, arr, clr) in zip(axes.flat, datasets_info):
    # Clip upper range for readability
    clip = np.percentile(arr, 99)
    clipped = arr[arr <= clip]

    ax.hist(clipped, bins=80, density=True, color=clr, alpha=0.65,
            edgecolor='white', linewidth=0.4, label='Observed')

    # Fitted normal
    mu, sigma = clipped.mean(), clipped.std(ddof=1)
    x = np.linspace(clipped.min(), clipped.max(), 300)
    ax.plot(x, sp_stats.norm.pdf(x, mu, sigma), color=NORMAL_CLR,
            linewidth=2.2, label=f'Normal fit (μ={mu:.0f}, σ={sigma:.0f})')

    # Best-fit log-normal
    shape, loc, scale = sp_stats.lognorm.fit(clipped, floc=0)
    ax.plot(x, sp_stats.lognorm.pdf(x, shape, loc, scale), color=VULN_CLR,
            linewidth=2.0, linestyle='--', label='Log-normal fit')

    ax.set_title(label, fontweight='bold')
    ax.set_ylabel('Density')
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.9, edgecolor='#CCC')

    # Annotation box
    desc = all_results[label]['_desc']
    txt = (f"skew = {desc['skew']:.2f}\n"
           f"kurtosis = {desc['kurtosis']:.2f}\n"
           f"n = {desc['n']:,}")
    ax.text(0.97, 0.60, txt, transform=ax.transAxes, fontsize=7.5,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='#F5F5F5', ec='#CCC'))

axes[1, 0].set_xlabel('Characters / Lines')
axes[1, 1].set_xlabel('Characters / Lines')

fig1.suptitle('Observed Distributions vs. Fitted Normal & Log-Normal Curves',
              fontsize=13, fontweight='bold', y=1.01)
fig1.tight_layout()
fig1.savefig('figures/normality_hist_overlay.pdf', dpi=300, bbox_inches='tight')
fig1.savefig('figures/normality_hist_overlay.png', dpi=300, bbox_inches='tight')
plt.close(fig1)
print("  → normality_hist_overlay.{pdf,png}")


# ============================================================================
# FIGURE 2 — Q-Q Plots (raw + log-transformed)  2 × 4
# ============================================================================
print("Figure 2: Q-Q plots (raw + log) …")

fig2, axes = plt.subplots(2, 4, figsize=(16, 8))

for col, (label, arr, clr) in enumerate(datasets_info):
    # Raw
    ax = axes[0, col]
    (osm, osr), (slope, intercept, r) = sp_stats.probplot(arr, dist='norm')
    ax.scatter(osm, osr, s=1.5, alpha=0.35, color=clr, rasterized=True)
    ax.plot(osm, slope * osm + intercept, color=NORMAL_CLR, linewidth=2,
            label=f'$R^2$ = {r**2:.4f}')
    ax.set_title(label, fontweight='bold', fontsize=9.5)
    ax.set_ylabel('Ordered Values')
    ax.legend(fontsize=7.5, loc='upper left')
    if col == 0:
        ax.set_ylabel('Raw\nOrdered Values', fontweight='bold')

    # Log-transformed
    ax = axes[1, col]
    log_arr = np.log1p(arr)
    (osm, osr), (slope, intercept, r) = sp_stats.probplot(log_arr, dist='norm')
    ax.scatter(osm, osr, s=1.5, alpha=0.35, color=clr, rasterized=True)
    ax.plot(osm, slope * osm + intercept, color=NORMAL_CLR, linewidth=2,
            label=f'$R^2$ = {r**2:.4f}')
    ax.set_title(f'{label}  (log₁ₚ)', fontweight='bold', fontsize=9.5)
    ax.set_xlabel('Theoretical Quantiles')
    ax.legend(fontsize=7.5, loc='upper left')
    if col == 0:
        ax.set_ylabel('log₁ₚ-Transformed\nOrdered Values', fontweight='bold')

fig2.suptitle('Normal Q-Q Plots — Raw vs. Log-Transformed',
              fontsize=13, fontweight='bold', y=1.01)
fig2.tight_layout()
fig2.savefig('figures/normality_qq_plots.pdf', dpi=300, bbox_inches='tight')
fig2.savefig('figures/normality_qq_plots.png', dpi=300, bbox_inches='tight')
plt.close(fig2)
print("  → normality_qq_plots.{pdf,png}")


# ============================================================================
# FIGURE 3 — Log-Transformed Histograms + Normal fit
# ============================================================================
print("Figure 3: Log-transformed histograms …")

fig3, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, (label, arr, clr) in zip(axes.flat, datasets_info):
    log_arr = np.log1p(arr)
    ax.hist(log_arr, bins=80, density=True, color=clr, alpha=0.65,
            edgecolor='white', linewidth=0.4, label='Observed (log₁ₚ)')

    mu, sigma = log_arr.mean(), log_arr.std(ddof=1)
    x = np.linspace(log_arr.min(), log_arr.max(), 300)
    ax.plot(x, sp_stats.norm.pdf(x, mu, sigma), color=NORMAL_CLR,
            linewidth=2.2, label=f'Normal fit (μ={mu:.2f}, σ={sigma:.2f})')

    ax.set_title(f'{label}  — log₁ₚ Transform', fontweight='bold')
    ax.set_ylabel('Density')
    ax.set_xlabel('log₁ₚ(value)')
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.9, edgecolor='#CCC')

    desc = all_results[label + ' (log)']['_desc']
    txt = (f"skew = {desc['skew']:.3f}\n"
           f"kurtosis = {desc['kurtosis']:.3f}")
    ax.text(0.97, 0.75, txt, transform=ax.transAxes, fontsize=8,
            ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='#F5F5F5', ec='#CCC'))

fig3.suptitle('Log-Transformed Distributions with Normal Overlay',
              fontsize=13, fontweight='bold', y=1.01)
fig3.tight_layout()
fig3.savefig('figures/normality_log_histograms.pdf', dpi=300, bbox_inches='tight')
fig3.savefig('figures/normality_log_histograms.png', dpi=300, bbox_inches='tight')
plt.close(fig3)
print("  → normality_log_histograms.{pdf,png}")


# ============================================================================
# FIGURE 4 — Statistical Test Summary Table
# ============================================================================
print("Figure 4: Statistical test summary table …")

fig4, ax4 = plt.subplots(figsize=(14, 7.5))
ax4.axis('off')

# Build table data
col_labels = ['Distribution', 'N', 'Skew', 'Kurtosis',
              'Shapiro–Wilk\n(stat / p)',
              "D'Agostino–\nPearson (p)",
              'Anderson–\nDarling (stat)',
              'A-D\n5% Crit.',
              'A-D\nReject?',
              'KS\n(stat / p)',
              'Normal?']

rows = []
for label, arr, clr in datasets_info:
    for suffix, display in [('', ''), (' (log)', '  (log₁ₚ)')]:
        key = label + suffix
        r = all_results[key]
        d = r['_desc']
        sw = r['Shapiro–Wilk']
        dag = r["D'Agostino–Pearson"]
        ad = r['Anderson–Darling']
        ks = r['KS']

        # Overall verdict
        reject_count = 0
        if sw['p'] < 0.05:  reject_count += 1
        if dag['p'] < 0.05: reject_count += 1
        if ad['reject']:    reject_count += 1
        if ks['p'] < 0.05:  reject_count += 1
        verdict = 'NO' if reject_count >= 3 else ('Marginal' if reject_count >= 2 else 'Yes')

        rows.append([
            label + display,
            f"{d['n']:,}",
            f"{d['skew']:.3f}",
            f"{d['kurtosis']:.3f}",
            f"{sw['stat']:.5f} / {sw['p']:.2e}",
            f"{dag['p']:.2e}",
            f"{ad['stat']:.3f}",
            f"{ad['crit_5pct']:.3f}",
            'Yes' if ad['reject'] else 'No',
            f"{ks['stat']:.4f} / {ks['p']:.2e}",
            verdict,
        ])

table = ax4.table(cellText=rows, colLabels=col_labels,
                  cellLoc='center', loc='center',
                  colWidths=[0.15, 0.06, 0.06, 0.06, 0.12, 0.08, 0.08, 0.06, 0.05, 0.12, 0.06])

table.auto_set_font_size(False)
table.set_fontsize(7.5)
table.scale(1.0, 1.7)

for (i, j), cell in table.get_celld().items():
    cell.set_edgecolor('#CCCCCC')
    cell.set_linewidth(0.5)
    if i == 0:
        cell.set_facecolor('#34495E')
        cell.set_text_props(color='white', fontweight='bold', fontsize=7)
    elif j == 0:
        cell.set_facecolor('#F8F9FA')
        cell.set_text_props(fontweight='bold', fontsize=7)
    else:
        cell.set_facecolor('white')
    # Colour the verdict column
    if i > 0 and j == 10:
        txt = cell.get_text().get_text()
        if txt == 'NO':
            cell.set_facecolor('#FADBD8')
            cell.set_text_props(fontweight='bold', color='#C0392B')
        elif txt == 'Marginal':
            cell.set_facecolor('#FEF9E7')
            cell.set_text_props(fontweight='bold', color='#D68910')
        else:
            cell.set_facecolor('#D5F5E3')
            cell.set_text_props(fontweight='bold', color='#27AE60')
    # Alternate row shading
    if i > 0 and i % 2 == 0 and j not in (0, 10):
        cell.set_facecolor('#FAFAFA')
    # Shade log rows slightly
    if i > 0 and '(log' in rows[i-1][0] and j not in (0, 10):
        base = '#F0F0F0' if i % 2 == 0 else '#F6F6F6'
        cell.set_facecolor(base)

ax4.set_title('Normality Test Battery — Raw & Log-Transformed Distributions\n'
              '(Significance level α = 0.05; Shapiro–Wilk uses n = 5 000 sub-sample)',
              fontsize=12, fontweight='bold', pad=18)

fig4.tight_layout()
fig4.savefig('figures/normality_test_table.pdf', dpi=300, bbox_inches='tight')
fig4.savefig('figures/normality_test_table.png', dpi=300, bbox_inches='tight')
plt.close(fig4)
print("  → normality_test_table.{pdf,png}")


# ============================================================================
# FIGURE 5 — Empirical CDF vs Theoretical Normal CDF
# ============================================================================
print("Figure 5: Empirical CDF vs Normal CDF …")

fig5, axes = plt.subplots(2, 2, figsize=(12, 9))

for row_idx, transform_label in enumerate(['Raw', 'log₁ₚ']):
    for col_idx, (label, arr, clr) in enumerate([
        ('Juliet C — Chars', juliet_chars, JULIET_CLR),
        ('Devign — Chars',   devign_chars, DEVIGN_CLR),
    ]):
        ax = axes[row_idx, col_idx]

        data = np.log1p(arr) if row_idx == 1 else arr.copy()
        # Clip for readability (raw only)
        if row_idx == 0:
            clip = np.percentile(data, 99)
            data = data[data <= clip]

        sorted_data = np.sort(data)
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        mu, sigma = sorted_data.mean(), sorted_data.std(ddof=1)
        theoretical = sp_stats.norm.cdf(sorted_data, mu, sigma)

        ax.plot(sorted_data, ecdf, color=clr, linewidth=1.8, label='Empirical CDF')
        ax.plot(sorted_data, theoretical, color=NORMAL_CLR, linewidth=1.8,
                linestyle='--', label='Normal CDF')

        # KS distance annotation
        ks_stat = np.max(np.abs(ecdf - theoretical))
        max_idx = np.argmax(np.abs(ecdf - theoretical))
        ax.annotate('', xy=(sorted_data[max_idx], ecdf[max_idx]),
                     xytext=(sorted_data[max_idx], theoretical[max_idx]),
                     arrowprops=dict(arrowstyle='<->', color=VULN_CLR, lw=1.5))
        ax.text(sorted_data[max_idx], (ecdf[max_idx] + theoretical[max_idx]) / 2,
                f' D = {ks_stat:.4f}', fontsize=7.5, color=VULN_CLR, fontweight='bold',
                va='center')

        suffix = '' if row_idx == 0 else ' (log₁ₚ)'
        ax.set_title(f'{label}{suffix}', fontweight='bold', fontsize=10)
        ax.set_ylabel('Cumulative Probability')
        ax.legend(fontsize=8, loc='lower right', framealpha=0.9, edgecolor='#CCC')
        if row_idx == 1:
            ax.set_xlabel('log₁ₚ(Characters)')
        else:
            ax.set_xlabel('Characters')

fig5.suptitle('Empirical vs. Theoretical Normal CDF\n(Kolmogorov–Smirnov distance D shown)',
              fontsize=13, fontweight='bold', y=1.02)
fig5.tight_layout()
fig5.savefig('figures/normality_cdf_comparison.pdf', dpi=300, bbox_inches='tight')
fig5.savefig('figures/normality_cdf_comparison.png', dpi=300, bbox_inches='tight')
plt.close(fig5)
print("  → normality_cdf_comparison.{pdf,png}")


# ============================================================================
# FIGURE 6 — Skewness & Kurtosis context plot + box plots
# ============================================================================
print("Figure 6: Skewness–Kurtosis diagram + box plots …")

fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(13, 6),
                                   gridspec_kw={'width_ratios': [1.2, 1]})

# ── Panel A: Cullen & Frey-style skewness–kurtosis map ──
ax = ax6a

# Reference regions for known distributions
# Normal: skew=0, excess kurtosis=0
ax.plot(0, 0, '*', markersize=16, color=NORMAL_CLR, zorder=10, label='Normal')

# Theoretical loci for reference distributions
# Exponential: skew=2, kurtosis=6
ax.plot(4, 6, 'D', markersize=10, color='#9B59B6', zorder=10, label='Exponential')
# Uniform: skew=0, kurtosis=-1.2
ax.plot(0, -1.2, 'P', markersize=10, color='#F1C40F', zorder=10, label='Uniform')

# Lognormal family locus (parametric curve)
log_sigmas = np.linspace(0.01, 2.5, 200)
ln_skew_sq = [(np.exp(s**2) + 2) ** 2 * (np.exp(s**2) - 1) for s in log_sigmas]
ln_kurtosis = [np.exp(4*s**2) + 2*np.exp(3*s**2) + 3*np.exp(2*s**2) - 6 for s in log_sigmas]
ax.plot(ln_skew_sq, ln_kurtosis, '-', color='#E74C3C', linewidth=1.5,
        alpha=0.6, label='Log-normal family', zorder=3)

# Gamma family
gamma_shapes = np.linspace(0.2, 100, 300)
gam_skew_sq = [(2.0 / np.sqrt(a)) ** 2 for a in gamma_shapes]
gam_kurt = [6.0 / a for a in gamma_shapes]
ax.plot(gam_skew_sq, gam_kurt, '--', color='#3498DB', linewidth=1.5,
        alpha=0.6, label='Gamma family', zorder=3)

# Plot our datasets
markers = ['o', 's', '^', 'v']
for idx, (label, arr, clr) in enumerate(datasets_info):
    sk = sp_stats.skew(arr)
    ku = sp_stats.kurtosis(arr)
    ax.plot(sk**2, ku, markers[idx], markersize=11, color=clr,
            markeredgecolor='#333', markeredgewidth=1.0,
            label=f'{label}', zorder=15)
    # log version
    log_arr = np.log1p(arr)
    sk_l = sp_stats.skew(log_arr)
    ku_l = sp_stats.kurtosis(log_arr)
    ax.plot(sk_l**2, ku_l, markers[idx], markersize=9, color=clr,
            markeredgecolor='#333', markeredgewidth=0.8,
            alpha=0.45, zorder=14)
    # Arrow from raw → log
    ax.annotate('', xy=(sk_l**2, ku_l), xytext=(sk**2, ku),
                arrowprops=dict(arrowstyle='->', color=clr, lw=1.2, alpha=0.5))

ax.set_xlabel('Squared Skewness', fontsize=10)
ax.set_ylabel('Excess Kurtosis', fontsize=10)
ax.set_title('(a)  Cullen–Frey Map\n(arrows: raw → log-transform)', fontweight='bold')
ax.legend(fontsize=7, loc='upper right', framealpha=0.9, edgecolor='#CCC', ncol=2)
ax.set_xlim(-0.5, max(12, ax.get_xlim()[1]))
ax.set_ylim(-2, max(20, ax.get_ylim()[1]))

# ── Panel B: Box plots (raw + log side by side) ──
ax = ax6b

bp_data = [juliet_chars, juliet_lines, devign_chars, devign_lines,
           np.log1p(juliet_chars), np.log1p(juliet_lines),
           np.log1p(devign_chars), np.log1p(devign_lines)]

positions = [1, 2, 3, 4, 6, 7, 8, 9]
colors_bp = [JULIET_CLR, JULIET_CLR, DEVIGN_CLR, DEVIGN_CLR] * 2

bp = ax.boxplot(bp_data, positions=positions, widths=0.55,
                patch_artist=True, showfliers=False,
                medianprops=dict(color='#222', linewidth=1.5),
                whiskerprops=dict(color='#666'),
                capprops=dict(color='#666'))

for patch, color in zip(bp['boxes'], colors_bp):
    patch.set_facecolor(color)
    patch.set_alpha(0.55)
    patch.set_edgecolor('#333')

ax.set_xticks([1, 2, 3, 4, 6, 7, 8, 9])
ax.set_xticklabels(['J-Chars', 'J-Lines', 'D-Chars', 'D-Lines',
                     'J-Chars\n(log)', 'J-Lines\n(log)', 'D-Chars\n(log)', 'D-Lines\n(log)'],
                    fontsize=7, rotation=30, ha='right')

ax.axvline(5, color='#CCC', linestyle=':', linewidth=1)
ax.text(2.5, ax.get_ylim()[1] * 0.95, 'Raw', ha='center', fontsize=9,
        fontweight='bold', color='#555')
ax.text(7.5, ax.get_ylim()[1] * 0.95, 'Log-transformed', ha='center', fontsize=9,
        fontweight='bold', color='#555')

ax.set_ylabel('Value')
ax.set_title('(b)  Box Plots (outliers hidden)', fontweight='bold')

fig6.suptitle('Distribution Shape Analysis',
              fontsize=13, fontweight='bold', y=1.01)
fig6.tight_layout()
fig6.savefig('figures/normality_skew_kurtosis.pdf', dpi=300, bbox_inches='tight')
fig6.savefig('figures/normality_skew_kurtosis.png', dpi=300, bbox_inches='tight')
plt.close(fig6)
print("  → normality_skew_kurtosis.{pdf,png}")


# ============================================================================
# FIGURE 7 — Per-class (vuln vs secure) normality Q-Q (chars only)
# ============================================================================
print("Figure 7: Per-class Q-Q plots …")

fig7, axes = plt.subplots(2, 4, figsize=(16, 8))

splits = [
    ('Juliet Vuln',   np.array([len(r[2]) for r in juliet if is_vuln(r[1])]),  VULN_CLR),
    ('Juliet Secure', np.array([len(r[2]) for r in juliet if not is_vuln(r[1])]), SEC_CLR),
    ('Devign Vuln',   np.array([len(r[2]) for r in devign if is_vuln(r[1])]),  VULN_CLR),
    ('Devign Secure', np.array([len(r[2]) for r in devign if not is_vuln(r[1])]), SEC_CLR),
]

for col, (lbl, arr, clr) in enumerate(splits):
    # Row 0: raw
    ax = axes[0, col]
    (osm, osr), (slope, intercept, r_val) = sp_stats.probplot(arr, dist='norm')
    ax.scatter(osm, osr, s=1.2, alpha=0.3, color=clr, rasterized=True)
    ax.plot(osm, slope * osm + intercept, color=NORMAL_CLR, lw=2,
            label=f'$R^2$ = {r_val**2:.4f}')
    ax.set_title(lbl, fontweight='bold', fontsize=9.5)
    ax.legend(fontsize=7, loc='upper left')
    if col == 0:
        ax.set_ylabel('Raw\nOrdered Values', fontweight='bold')
    desc = normality_battery(arr, lbl)['_desc']
    ax.text(0.97, 0.15, f"n={desc['n']:,}\nskew={desc['skew']:.2f}",
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', fc='#F5F5F5', ec='#CCC'))

    # Row 1: log
    ax = axes[1, col]
    log_arr = np.log1p(arr)
    (osm, osr), (slope, intercept, r_val) = sp_stats.probplot(log_arr, dist='norm')
    ax.scatter(osm, osr, s=1.2, alpha=0.3, color=clr, rasterized=True)
    ax.plot(osm, slope * osm + intercept, color=NORMAL_CLR, lw=2,
            label=f'$R^2$ = {r_val**2:.4f}')
    ax.set_title(f'{lbl} (log₁ₚ)', fontweight='bold', fontsize=9.5)
    ax.set_xlabel('Theoretical Quantiles')
    ax.legend(fontsize=7, loc='upper left')
    if col == 0:
        ax.set_ylabel('log₁ₚ\nOrdered Values', fontweight='bold')

fig7.suptitle('Normal Q-Q Plots by Vulnerability Class (Characters per Function)',
              fontsize=13, fontweight='bold', y=1.01)
fig7.tight_layout()
fig7.savefig('figures/normality_qq_by_class.pdf', dpi=300, bbox_inches='tight')
fig7.savefig('figures/normality_qq_by_class.png', dpi=300, bbox_inches='tight')
plt.close(fig7)
print("  → normality_qq_by_class.{pdf,png}")


# ============================================================================
# Print text summary for paper reference
# ============================================================================
print("\n" + "=" * 70)
print("NORMALITY TEST SUMMARY FOR PAPER")
print("=" * 70)
for label, arr, _ in datasets_info:
    for suffix, disp in [('', ''), (' (log)', ' [log₁ₚ]')]:
        key = label + suffix
        r = all_results[key]
        d = r['_desc']
        sw = r['Shapiro–Wilk']
        dag = r["D'Agostino–Pearson"]
        ad = r['Anderson–Darling']
        ks = r['KS']
        print(f"\n{label}{disp}:")
        print(f"  N={d['n']:,}  mean={d['mean']:.1f}  std={d['std']:.1f}  "
              f"skew={d['skew']:.3f}  kurtosis={d['kurtosis']:.3f}")
        print(f"  Shapiro–Wilk:     W={sw['stat']:.5f}  p={sw['p']:.3e}  (n={sw['n_used']})")
        print(f"  D'Agostino:       p={dag['p']:.3e}")
        print(f"  Anderson–Darling: A²={ad['stat']:.3f}  crit(5%)={ad['crit_5pct']:.3f}  "
              f"reject={'Yes' if ad['reject'] else 'No'}")
        print(f"  KS:               D={ks['stat']:.4f}  p={ks['p']:.3e}")

print("\n" + "=" * 70)
print("All normality figures generated!")
print("=" * 70)
print("\nFiles:")
for f in sorted(os.listdir('figures')):
    if 'normality' in f and f.endswith(('.pdf', '.png')):
        size = os.path.getsize(os.path.join('figures', f)) / 1024
        print(f"  {f:<50} {size:>7.1f} KB")
