"""
Publication-Quality Dataset Analysis Figures
=============================================
Generates five figures characterising the Juliet C and Devign datasets:

  1. CWE distribution (Juliet C) — horizontal bar chart
  2. Class balance comparison — grouped bar chart
  3. Code length / token length distributions — dual histograms
  4. SimHash deduplication curve — lines + retained % annotation
  5. Devign per-project breakdown — stacked bar + pie

Outputs saved to figures/ as both PDF and PNG (300 DPI).
"""

import sqlite3
import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
from collections import Counter, defaultdict

# ── Style ─────────────────────────────────────────────────────────────────────
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

JULIET_DB = 'datasets/juliet_c.db'
DEVIGN_DB = 'datasets/devign.db'
SIMHASH_DIR = 'simhash_datasets'

# ── Colour palette ────────────────────────────────────────────────────────────
VULN_COLOR    = '#C0392B'   # red
SECURE_COLOR  = '#2E86AB'   # blue
JULIET_COLOR  = '#E67E22'   # orange
DEVIGN_COLOR  = '#2980B9'   # blue
FFMPEG_COLOR  = '#27AE60'   # green
QEMU_COLOR    = '#8E44AD'   # purple
ACCENT        = '#34495E'


# ============================================================================
# Helper: load all functions from a database
# ============================================================================
def load_funcs(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT grp, id, start, end, vuln, code, len FROM funcs")
    rows = cur.fetchall()
    conn.close()
    return rows

def is_vulnerable(vuln):
    if vuln is None:
        return False
    v = str(vuln).strip()
    return v != '' and v != '0'


# ============================================================================
# Load data
# ============================================================================
print("Loading databases...")
juliet_rows = load_funcs(JULIET_DB)
devign_rows = load_funcs(DEVIGN_DB)

print(f"  Juliet C: {len(juliet_rows):,} functions")
print(f"  Devign:   {len(devign_rows):,} functions")


# ============================================================================
# FIGURE 1 — CWE Distribution (Juliet C), Top 25
# ============================================================================
print("\nGenerating Figure 1: CWE Distribution...")

cwe_vuln = Counter()
cwe_secure = Counter()
for grp, fid, start, end, vuln, code, ln in juliet_rows:
    cwe = re.match(r'(CWE\d+)', str(grp))
    cwe = cwe.group(1) if cwe else 'Other'
    if is_vulnerable(vuln):
        cwe_vuln[cwe] += 1
    else:
        cwe_secure[cwe] += 1

all_cwes = sorted(set(cwe_vuln) | set(cwe_secure),
                  key=lambda c: cwe_vuln[c] + cwe_secure[c], reverse=True)
top_n = 25
top_cwes = all_cwes[:top_n]

fig1, ax1 = plt.subplots(figsize=(8, 7))

y_pos = np.arange(len(top_cwes))
vuln_vals   = [cwe_vuln[c] for c in top_cwes]
secure_vals = [cwe_secure[c] for c in top_cwes]

bars_v = ax1.barh(y_pos, vuln_vals, height=0.40, align='center',
                  color=VULN_COLOR, alpha=0.85, label='Vulnerable', edgecolor='white', linewidth=0.5)
bars_s = ax1.barh(y_pos, [-s for s in secure_vals], height=0.40, align='center',
                  color=SECURE_COLOR, alpha=0.85, label='Non-Vulnerable', edgecolor='white', linewidth=0.5)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(top_cwes, fontsize=8)
ax1.invert_yaxis()

# Symmetric x-axis
xmax = max(max(vuln_vals), max(secure_vals)) * 1.1
ax1.set_xlim(-xmax, xmax)

# Custom x tick labels (absolute values)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{abs(int(x)):,}'))

ax1.set_xlabel('Number of Functions')
ax1.set_title(f'Juliet C/C++ Test Suite — CWE Distribution (Top {top_n} of {len(all_cwes)})',
              fontsize=12, fontweight='bold', pad=12)
ax1.axvline(0, color='#333', linewidth=0.8)
ax1.legend(loc='lower right', framealpha=0.9, edgecolor='#CCC')

# Annotation: total
ax1.text(0.98, 0.02,
         f'Total: {len(juliet_rows):,} functions\n{len(all_cwes)} distinct CWEs\n'
         f'{sum(cwe_vuln.values()):,} vuln / {sum(cwe_secure.values()):,} secure',
         transform=ax1.transAxes, fontsize=8, ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', edgecolor='#CCC'))

fig1.tight_layout()
fig1.savefig('figures/juliet_c_cwe_distribution.pdf', dpi=300, bbox_inches='tight')
fig1.savefig('figures/juliet_c_cwe_distribution.png', dpi=300, bbox_inches='tight')
plt.close(fig1)
print("  Saved: juliet_c_cwe_distribution.{pdf,png}")


# ============================================================================
# FIGURE 2 — Class Balance Comparison (both datasets)
# ============================================================================
print("Generating Figure 2: Class Balance...")

juliet_v  = sum(1 for r in juliet_rows if is_vulnerable(r[4]))
juliet_nv = len(juliet_rows) - juliet_v

devign_v  = sum(1 for r in devign_rows if is_vulnerable(r[4]))
devign_nv = len(devign_rows) - devign_v

# Devign per project
devign_proj = defaultdict(lambda: {'vuln': 0, 'secure': 0})
for grp, fid, start, end, vuln, code, ln in devign_rows:
    if is_vulnerable(vuln):
        devign_proj[grp]['vuln'] += 1
    else:
        devign_proj[grp]['secure'] += 1

fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4.5), gridspec_kw={'width_ratios': [1, 1, 1.3]})

# Panel A: Juliet C
ax = axes2[0]
bars = ax.bar(['Vulnerable', 'Non-Vuln.'], [juliet_v, juliet_nv],
              color=[VULN_COLOR, SECURE_COLOR], alpha=0.85, edgecolor='white', width=0.55)
for bar, val in zip(bars, [juliet_v, juliet_nv]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1500,
            f'{val:,}\n({val/len(juliet_rows)*100:.1f}%)',
            ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.set_ylabel('Number of Functions')
ax.set_title('(a)  Juliet C/C++', fontsize=10.5, fontweight='bold')
ax.set_ylim(0, max(juliet_v, juliet_nv) * 1.2)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

# Panel B: Devign
ax = axes2[1]
bars = ax.bar(['Vulnerable', 'Non-Vuln.'], [devign_v, devign_nv],
              color=[VULN_COLOR, SECURE_COLOR], alpha=0.85, edgecolor='white', width=0.55)
for bar, val in zip(bars, [devign_v, devign_nv]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{val:,}\n({val/len(devign_rows)*100:.1f}%)',
            ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax.set_title('(b)  Devign', fontsize=10.5, fontweight='bold')
ax.set_ylim(0, max(devign_v, devign_nv) * 1.2)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

# Panel C: Devign per-project
ax = axes2[2]
projects = sorted(devign_proj.keys())
x = np.arange(len(projects))
vuln_vals = [devign_proj[p]['vuln'] for p in projects]
sec_vals  = [devign_proj[p]['secure'] for p in projects]
w = 0.35
b1 = ax.bar(x - w/2, vuln_vals, w, label='Vulnerable', color=VULN_COLOR, alpha=0.85, edgecolor='white')
b2 = ax.bar(x + w/2, sec_vals, w, label='Non-Vulnerable', color=SECURE_COLOR, alpha=0.85, edgecolor='white')
for bars_set, vals in [(b1, vuln_vals), (b2, sec_vals)]:
    for bar, val in zip(bars_set, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
                f'{val:,}', ha='center', va='bottom', fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(projects, fontsize=9)
ax.set_title('(c)  Devign by Project', fontsize=10.5, fontweight='bold')
ax.legend(loc='upper right', framealpha=0.9, edgecolor='#CCC')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

fig2.suptitle('Dataset Class Balance', fontsize=13, fontweight='bold', y=1.02)
fig2.tight_layout()
fig2.savefig('figures/dataset_class_balance.pdf', dpi=300, bbox_inches='tight')
fig2.savefig('figures/dataset_class_balance.png', dpi=300, bbox_inches='tight')
plt.close(fig2)
print("  Saved: dataset_class_balance.{pdf,png}")


# ============================================================================
# FIGURE 3 — Code Length Distributions (chars + lines)
# ============================================================================
print("Generating Figure 3: Code Length Distributions...")

juliet_char_lens = np.array([len(r[5]) for r in juliet_rows])
devign_char_lens = np.array([len(r[5]) for r in devign_rows])

juliet_line_lens = np.array([r[5].count('\n') + 1 for r in juliet_rows])
devign_line_lens = np.array([r[5].count('\n') + 1 for r in devign_rows])

fig3, axes3 = plt.subplots(2, 2, figsize=(12, 8))

# 3a: Juliet — character length
ax = axes3[0, 0]
ax.hist(juliet_char_lens, bins=80, color=JULIET_COLOR, alpha=0.8,
        edgecolor='white', linewidth=0.4, range=(0, 5000))
ax.axvline(np.median(juliet_char_lens), color='#222', linestyle='--', lw=1.2, alpha=0.7)
ax.text(np.median(juliet_char_lens) + 80, ax.get_ylim()[1]*0.90,
        f'median = {np.median(juliet_char_lens):.0f}',
        fontsize=8, color='#222')
ax.set_title('(a)  Juliet C — Characters per Function', fontweight='bold')
ax.set_xlabel('Characters')
ax.set_ylabel('Count')
# Stats box
stats_text = (f'N = {len(juliet_char_lens):,}\n'
              f'mean = {juliet_char_lens.mean():.0f}\n'
              f'median = {np.median(juliet_char_lens):.0f}\n'
              f'p95 = {np.percentile(juliet_char_lens, 95):.0f}')
ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=7.5,
        ha='right', va='top', bbox=dict(boxstyle='round,pad=0.4', fc='#FFF8E1', ec='#E67E22', alpha=0.9))

# 3b: Devign — character length
ax = axes3[0, 1]
ax.hist(devign_char_lens, bins=80, color=DEVIGN_COLOR, alpha=0.8,
        edgecolor='white', linewidth=0.4, range=(0, 12000))
ax.axvline(np.median(devign_char_lens), color='#222', linestyle='--', lw=1.2, alpha=0.7)
ax.text(np.median(devign_char_lens) + 200, ax.get_ylim()[1]*0.90,
        f'median = {np.median(devign_char_lens):.0f}',
        fontsize=8, color='#222')
ax.set_title('(b)  Devign — Characters per Function', fontweight='bold')
ax.set_xlabel('Characters')
ax.set_ylabel('Count')
stats_text = (f'N = {len(devign_char_lens):,}\n'
              f'mean = {devign_char_lens.mean():.0f}\n'
              f'median = {np.median(devign_char_lens):.0f}\n'
              f'p95 = {np.percentile(devign_char_lens, 95):.0f}')
ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=7.5,
        ha='right', va='top', bbox=dict(boxstyle='round,pad=0.4', fc='#E3F2FD', ec='#2980B9', alpha=0.9))

# 3c: Juliet — line count
ax = axes3[1, 0]
ax.hist(juliet_line_lens, bins=80, color=JULIET_COLOR, alpha=0.8,
        edgecolor='white', linewidth=0.4, range=(0, 150))
ax.axvline(np.median(juliet_line_lens), color='#222', linestyle='--', lw=1.2, alpha=0.7)
ax.text(np.median(juliet_line_lens) + 3, ax.get_ylim()[1]*0.90,
        f'median = {np.median(juliet_line_lens):.0f}',
        fontsize=8, color='#222')
ax.set_title('(c)  Juliet C — Lines per Function', fontweight='bold')
ax.set_xlabel('Lines of Code')
ax.set_ylabel('Count')
stats_text = (f'mean = {juliet_line_lens.mean():.1f}\n'
              f'median = {np.median(juliet_line_lens):.0f}\n'
              f'p95 = {np.percentile(juliet_line_lens, 95):.0f}')
ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=7.5,
        ha='right', va='top', bbox=dict(boxstyle='round,pad=0.4', fc='#FFF8E1', ec='#E67E22', alpha=0.9))

# 3d: Devign — line count
ax = axes3[1, 1]
ax.hist(devign_line_lens, bins=80, color=DEVIGN_COLOR, alpha=0.8,
        edgecolor='white', linewidth=0.4, range=(0, 300))
ax.axvline(np.median(devign_line_lens), color='#222', linestyle='--', lw=1.2, alpha=0.7)
ax.text(np.median(devign_line_lens) + 5, ax.get_ylim()[1]*0.90,
        f'median = {np.median(devign_line_lens):.0f}',
        fontsize=8, color='#222')
ax.set_title('(d)  Devign — Lines per Function', fontweight='bold')
ax.set_xlabel('Lines of Code')
ax.set_ylabel('Count')
stats_text = (f'mean = {devign_line_lens.mean():.1f}\n'
              f'median = {np.median(devign_line_lens):.0f}\n'
              f'p95 = {np.percentile(devign_line_lens, 95):.0f}')
ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=7.5,
        ha='right', va='top', bbox=dict(boxstyle='round,pad=0.4', fc='#E3F2FD', ec='#2980B9', alpha=0.9))

fig3.suptitle('Function-Level Size Distributions', fontsize=13, fontweight='bold', y=1.01)
fig3.tight_layout()
fig3.savefig('figures/code_length_distributions.pdf', dpi=300, bbox_inches='tight')
fig3.savefig('figures/code_length_distributions.png', dpi=300, bbox_inches='tight')
plt.close(fig3)
print("  Saved: code_length_distributions.{pdf,png}")


# ============================================================================
# FIGURE 4 — SimHash Deduplication Curve
# ============================================================================
print("Generating Figure 4: SimHash Deduplication Curve...")

def get_simhash_counts(dataset_prefix, k_range=range(1, 16)):
    counts = {}
    for k in k_range:
        db_path = os.path.join(SIMHASH_DIR, f'{dataset_prefix}_simhash_k={k}.db')
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute('SELECT COUNT(*) FROM funcs')
            total = cur.fetchone()[0]
            cur.execute("""SELECT 
                SUM(CASE WHEN vuln IS NOT NULL AND vuln != '' AND vuln != '0' THEN 1 ELSE 0 END),
                SUM(CASE WHEN vuln IS NULL OR vuln = '' OR vuln = '0' THEN 1 ELSE 0 END)
                FROM funcs""")
            v, nv = cur.fetchone()
            conn.close()
            counts[k] = {'total': total, 'vuln': v, 'secure': nv}
    return counts

juliet_simhash = get_simhash_counts('juliet_c')
devign_simhash = get_simhash_counts('devign')

# Get original (pre-dedup) counts
juliet_orig = len(juliet_rows)
devign_orig = len(devign_rows)

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(13, 5.5))

# ── Panel A: Absolute counts ──
k_vals_j = sorted(juliet_simhash.keys())
k_vals_d = sorted(devign_simhash.keys())

ax = ax4a
ax.plot(k_vals_j, [juliet_simhash[k]['total'] for k in k_vals_j],
        'o-', color=JULIET_COLOR, linewidth=2.2, markersize=5, label='Juliet C — Total', zorder=5)
ax.plot(k_vals_j, [juliet_simhash[k]['vuln'] for k in k_vals_j],
        's--', color=JULIET_COLOR, linewidth=1.4, markersize=4, alpha=0.65, label='Juliet C — Vulnerable')
ax.plot(k_vals_j, [juliet_simhash[k]['secure'] for k in k_vals_j],
        '^--', color=JULIET_COLOR, linewidth=1.4, markersize=4, alpha=0.45, label='Juliet C — Secure')

ax.plot(k_vals_d, [devign_simhash[k]['total'] for k in k_vals_d],
        'o-', color=DEVIGN_COLOR, linewidth=2.2, markersize=5, label='Devign — Total', zorder=5)
ax.plot(k_vals_d, [devign_simhash[k]['vuln'] for k in k_vals_d],
        's--', color=DEVIGN_COLOR, linewidth=1.4, markersize=4, alpha=0.65, label='Devign — Vulnerable')
ax.plot(k_vals_d, [devign_simhash[k]['secure'] for k in k_vals_d],
        '^--', color=DEVIGN_COLOR, linewidth=1.4, markersize=4, alpha=0.45, label='Devign — Secure')

ax.set_xlabel('SimHash Hamming Distance Threshold ($k$)')
ax.set_ylabel('Number of Functions')
ax.set_title('(a)  Absolute Function Counts', fontsize=10.5, fontweight='bold')
ax.legend(loc='upper right', fontsize=7.5, framealpha=0.9, edgecolor='#CCC', ncol=2)
ax.set_xticks(range(1, 16))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

# Shade k=1–12 region (used in experiments)
ax.axvspan(0.5, 12.5, alpha=0.06, color='#27AE60')
ax.text(6.5, ax.get_ylim()[1]*0.97, 'Experimental range (k = 1–12)',
        fontsize=7.5, ha='center', va='top', color='#27AE60', fontstyle='italic')

# ── Panel B: Percentage retained ──
ax = ax4b
juliet_pct = [juliet_simhash[k]['total'] / juliet_orig * 100 for k in k_vals_j]
devign_pct = [devign_simhash[k]['total'] / devign_orig * 100 for k in k_vals_d]

ax.plot(k_vals_j, juliet_pct,
        'o-', color=JULIET_COLOR, linewidth=2.2, markersize=6, label='Juliet C', zorder=5)
ax.plot(k_vals_d, devign_pct,
        's-', color=DEVIGN_COLOR, linewidth=2.2, markersize=6, label='Devign', zorder=5)

# Fill between to emphasize gap
ax.fill_between(k_vals_j[:min(len(k_vals_j), len(k_vals_d))],
                juliet_pct[:min(len(k_vals_j), len(k_vals_d))],
                devign_pct[:min(len(k_vals_j), len(k_vals_d))],
                alpha=0.08, color='#888')

# Annotate key points
for k_ann in [1, 6, 12]:
    if k_ann in juliet_simhash:
        j_pct = juliet_simhash[k_ann]['total'] / juliet_orig * 100
        ax.annotate(f'{j_pct:.1f}%', (k_ann, j_pct), textcoords='offset points',
                    xytext=(-18, 10), fontsize=7, color=JULIET_COLOR, fontweight='bold')
    if k_ann in devign_simhash:
        d_pct = devign_simhash[k_ann]['total'] / devign_orig * 100
        ax.annotate(f'{d_pct:.1f}%', (k_ann, d_pct), textcoords='offset points',
                    xytext=(5, -15), fontsize=7, color=DEVIGN_COLOR, fontweight='bold')

ax.set_xlabel('SimHash Hamming Distance Threshold ($k$)')
ax.set_ylabel('Percentage of Original Dataset Retained (%)')
ax.set_title('(b)  Data Retention Under Deduplication', fontsize=10.5, fontweight='bold')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9, edgecolor='#CCC')
ax.set_xticks(range(1, 16))
ax.set_ylim(0, 105)
ax.axvspan(0.5, 12.5, alpha=0.06, color='#27AE60')

# Key takeaway annotation
ax.text(0.03, 0.05,
        'Juliet C loses data much faster\n'
        'than Devign, indicating far\n'
        'higher levels of near-duplication\n'
        'in the synthetic benchmark.',
        transform=ax.transAxes, fontsize=7.5, va='bottom', ha='left',
        bbox=dict(boxstyle='round,pad=0.5', fc='#FFF8E1', ec='#E67E22', alpha=0.9))

fig4.suptitle('Impact of SimHash Deduplication on Dataset Size',
              fontsize=13, fontweight='bold', y=1.02)
fig4.tight_layout()
fig4.savefig('figures/simhash_deduplication_curve.pdf', dpi=300, bbox_inches='tight')
fig4.savefig('figures/simhash_deduplication_curve.png', dpi=300, bbox_inches='tight')
plt.close(fig4)
print("  Saved: simhash_deduplication_curve.{pdf,png}")


# ============================================================================
# FIGURE 5 — Comprehensive Summary Table Figure
# ============================================================================
print("Generating Figure 5: Comparative Summary Table...")

fig5, ax5 = plt.subplots(figsize=(10, 4.5))
ax5.axis('off')

# Build summary data
summary_data = [
    ['', 'Juliet C/C++', 'Devign'],
    ['Source', 'NSA SARD (synthetic)', 'FFmpeg + QEMU (real-world)'],
    ['Language', 'C / C++', 'C'],
    ['Total Functions', f'{len(juliet_rows):,}', f'{len(devign_rows):,}'],
    ['Vulnerable', f'{juliet_v:,} ({juliet_v/len(juliet_rows)*100:.1f}%)',
                   f'{devign_v:,} ({devign_v/len(devign_rows)*100:.1f}%)'],
    ['Non-Vulnerable', f'{juliet_nv:,} ({juliet_nv/len(juliet_rows)*100:.1f}%)',
                       f'{devign_nv:,} ({devign_nv/len(devign_rows)*100:.1f}%)'],
    ['Distinct CWEs / Projects', f'{len(all_cwes)}', '2 (FFmpeg, QEMU)'],
    ['Median Chars/Function', f'{np.median(juliet_char_lens):.0f}',
                               f'{np.median(devign_char_lens):.0f}'],
    ['Mean Chars/Function', f'{juliet_char_lens.mean():.0f}',
                             f'{devign_char_lens.mean():.0f}'],
    ['P95 Chars/Function', f'{np.percentile(juliet_char_lens, 95):.0f}',
                            f'{np.percentile(devign_char_lens, 95):.0f}'],
    ['Median Lines/Function', f'{np.median(juliet_line_lens):.0f}',
                               f'{np.median(devign_line_lens):.0f}'],
    ['Mean Lines/Function', f'{juliet_line_lens.mean():.1f}',
                             f'{devign_line_lens.mean():.1f}'],
    ['After SimHash k=1', f'{juliet_simhash[1]["total"]:,} ({juliet_simhash[1]["total"]/juliet_orig*100:.1f}%)',
                          f'{devign_simhash[1]["total"]:,} ({devign_simhash[1]["total"]/devign_orig*100:.1f}%)'],
    ['After SimHash k=12', f'{juliet_simhash[12]["total"]:,} ({juliet_simhash[12]["total"]/juliet_orig*100:.1f}%)',
                           f'{devign_simhash[12]["total"]:,} ({devign_simhash[12]["total"]/devign_orig*100:.1f}%)'],
    ['Experimental Role', 'Train (Exp 1–3)\nOOD Eval (Exp 4)', 'OOD Eval (Exp 1–3)\nTrain (Exp 4)'],
]

table = ax5.table(
    cellText=[row for row in summary_data[1:]],
    colLabels=summary_data[0],
    cellLoc='center',
    loc='center',
    colWidths=[0.28, 0.36, 0.36],
)

table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1.0, 1.55)

# Style
for (i, j), cell in table.get_celld().items():
    cell.set_edgecolor('#CCCCCC')
    cell.set_linewidth(0.5)
    if i == 0:  # header
        cell.set_facecolor('#34495E')
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)
        cell.set_height(0.08)
    elif j == 0:  # row labels
        cell.set_facecolor('#F8F9FA')
        cell.set_text_props(fontweight='bold', fontsize=8.5)
    else:
        cell.set_facecolor('white')
    # Alternate row shading
    if i > 0 and i % 2 == 0 and j != 0:
        cell.set_facecolor('#FAFAFA')

ax5.set_title('Dataset Summary Statistics',
              fontsize=13, fontweight='bold', pad=15)

fig5.tight_layout()
fig5.savefig('figures/dataset_summary_table.pdf', dpi=300, bbox_inches='tight')
fig5.savefig('figures/dataset_summary_table.png', dpi=300, bbox_inches='tight')
plt.close(fig5)
print("  Saved: dataset_summary_table.{pdf,png}")


# ============================================================================
# FIGURE 6 — Side-by-side violin plot of code length
# ============================================================================
print("Generating Figure 6: Violin Plots...")

fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(12, 5))

# Split by vuln/secure for each dataset
juliet_vuln_chars  = [len(r[5]) for r in juliet_rows if is_vulnerable(r[4])]
juliet_sec_chars   = [len(r[5]) for r in juliet_rows if not is_vulnerable(r[4])]
devign_vuln_chars  = [len(r[5]) for r in devign_rows if is_vulnerable(r[4])]
devign_sec_chars   = [len(r[5]) for r in devign_rows if not is_vulnerable(r[4])]

# Panel A: Characters
ax = ax6a
parts_data = [juliet_vuln_chars, juliet_sec_chars, devign_vuln_chars, devign_sec_chars]
positions = [1, 2, 4, 5]
colors = [VULN_COLOR, SECURE_COLOR, VULN_COLOR, SECURE_COLOR]
labels_x = ['Vuln', 'Secure', 'Vuln', 'Secure']

vp = ax.violinplot(parts_data, positions=positions, showmeans=True,
                   showmedians=True, showextrema=False)

for i, body in enumerate(vp['bodies']):
    body.set_facecolor(colors[i])
    body.set_alpha(0.6)
    body.set_edgecolor(colors[i])
    body.set_linewidth(1.2)

vp['cmeans'].set_color('#333')
vp['cmeans'].set_linewidth(1.5)
vp['cmedians'].set_color('#E67E22')
vp['cmedians'].set_linewidth(1.5)

ax.set_xticks(positions)
ax.set_xticklabels(labels_x, fontsize=9)
ax.set_ylim(0, 8000)
ax.set_ylabel('Characters per Function')
ax.set_title('(a)  Character Length by Vulnerability Status', fontweight='bold')

# Group labels
ax.text(1.5, ax.get_ylim()[1]*0.96, 'Juliet C', ha='center', fontsize=10,
        fontweight='bold', color=JULIET_COLOR)
ax.text(4.5, ax.get_ylim()[1]*0.96, 'Devign', ha='center', fontsize=10,
        fontweight='bold', color=DEVIGN_COLOR)
ax.axvline(3, color='#CCCCCC', linestyle=':', linewidth=1)

# Panel B: Lines
juliet_vuln_lines  = [r[5].count('\n')+1 for r in juliet_rows if is_vulnerable(r[4])]
juliet_sec_lines   = [r[5].count('\n')+1 for r in juliet_rows if not is_vulnerable(r[4])]
devign_vuln_lines  = [r[5].count('\n')+1 for r in devign_rows if is_vulnerable(r[4])]
devign_sec_lines   = [r[5].count('\n')+1 for r in devign_rows if not is_vulnerable(r[4])]

ax = ax6b
parts_data = [juliet_vuln_lines, juliet_sec_lines, devign_vuln_lines, devign_sec_lines]

vp = ax.violinplot(parts_data, positions=positions, showmeans=True,
                   showmedians=True, showextrema=False)

for i, body in enumerate(vp['bodies']):
    body.set_facecolor(colors[i])
    body.set_alpha(0.6)
    body.set_edgecolor(colors[i])
    body.set_linewidth(1.2)

vp['cmeans'].set_color('#333')
vp['cmeans'].set_linewidth(1.5)
vp['cmedians'].set_color('#E67E22')
vp['cmedians'].set_linewidth(1.5)

ax.set_xticks(positions)
ax.set_xticklabels(labels_x, fontsize=9)
ax.set_ylim(0, 250)
ax.set_ylabel('Lines per Function')
ax.set_title('(b)  Line Count by Vulnerability Status', fontweight='bold')

ax.text(1.5, ax.get_ylim()[1]*0.96, 'Juliet C', ha='center', fontsize=10,
        fontweight='bold', color=JULIET_COLOR)
ax.text(4.5, ax.get_ylim()[1]*0.96, 'Devign', ha='center', fontsize=10,
        fontweight='bold', color=DEVIGN_COLOR)
ax.axvline(3, color='#CCCCCC', linestyle=':', linewidth=1)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#333', linewidth=1.5, label='Mean'),
    Line2D([0], [0], color='#E67E22', linewidth=1.5, label='Median'),
]
ax6b.legend(handles=legend_elements, loc='upper right', framealpha=0.9, edgecolor='#CCC')

fig6.suptitle('Function Size Distributions by Vulnerability Label',
              fontsize=13, fontweight='bold', y=1.01)
fig6.tight_layout()
fig6.savefig('figures/size_by_vulnerability_violins.pdf', dpi=300, bbox_inches='tight')
fig6.savefig('figures/size_by_vulnerability_violins.png', dpi=300, bbox_inches='tight')
plt.close(fig6)
print("  Saved: size_by_vulnerability_violins.{pdf,png}")


# ============================================================================
# FIGURE 7 — Juliet C CWE Bubble Chart (functions vs. vuln ratio, sized by N)
# ============================================================================
print("Generating Figure 7: CWE Bubble Chart...")

fig7, ax7 = plt.subplots(figsize=(10, 7))

cwe_names = []
cwe_totals = []
cwe_vuln_ratios = []
cwe_median_lens = []

# Pre-compute per-CWE character lengths
cwe_char_lens_map = defaultdict(list)
for grp, fid, start, end, vuln, code, ln in juliet_rows:
    cwe = re.match(r'(CWE\d+)', str(grp))
    cwe = cwe.group(1) if cwe else 'Other'
    cwe_char_lens_map[cwe].append(len(code))

for cwe in all_cwes:
    total = cwe_vuln[cwe] + cwe_secure[cwe]
    if total < 50:
        continue
    cwe_names.append(cwe)
    cwe_totals.append(total)
    cwe_vuln_ratios.append(cwe_vuln[cwe] / total)
    cwe_median_lens.append(np.median(cwe_char_lens_map[cwe]))

cwe_totals = np.array(cwe_totals)
cwe_vuln_ratios = np.array(cwe_vuln_ratios)
cwe_median_lens = np.array(cwe_median_lens)

# Size by total, color by median length
sizes = np.sqrt(cwe_totals) * 3

scatter = ax7.scatter(cwe_median_lens, cwe_vuln_ratios,
                      s=sizes, c=cwe_median_lens,
                      cmap='YlOrRd', alpha=0.7, edgecolors='#333', linewidths=0.5)

# Label the largest CWEs
for i in np.argsort(-cwe_totals)[:15]:
    ax7.annotate(cwe_names[i],
                 (cwe_median_lens[i], cwe_vuln_ratios[i]),
                 textcoords='offset points', xytext=(6, 4),
                 fontsize=6.5, color='#333', alpha=0.9)

cbar = plt.colorbar(scatter, ax=ax7, shrink=0.8, pad=0.02)
cbar.set_label('Median Character Length', fontsize=9)

ax7.set_xlabel('Median Function Length (characters)')
ax7.set_ylabel('Vulnerable Fraction')
ax7.set_title('Juliet C — CWE Characteristics\n(bubble size ∝ total functions, colour ∝ median length)',
              fontsize=11, fontweight='bold')
ax7.set_ylim(0.15, 0.85)
ax7.axhline(0.5, color='#888', linestyle=':', alpha=0.5, linewidth=1)
ax7.text(ax7.get_xlim()[1]*0.98, 0.505, 'balanced',
         fontsize=7, color='#888', ha='right', va='bottom', fontstyle='italic')

fig7.tight_layout()
fig7.savefig('figures/juliet_c_cwe_bubble_chart.pdf', dpi=300, bbox_inches='tight')
fig7.savefig('figures/juliet_c_cwe_bubble_chart.png', dpi=300, bbox_inches='tight')
plt.close(fig7)
print("  Saved: juliet_c_cwe_bubble_chart.{pdf,png}")


# ============================================================================
# Done
# ============================================================================
print("\n" + "=" * 60)
print("All figures generated successfully!")
print("=" * 60)
print("Files in figures/:")
for f in sorted(os.listdir('figures')):
    if f.endswith(('.pdf', '.png')):
        fpath = os.path.join('figures', f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f:<45} {size_kb:>7.1f} KB")
