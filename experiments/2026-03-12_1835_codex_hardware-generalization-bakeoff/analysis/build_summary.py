import csv, json, math
from pathlib import Path
from statistics import mean
import matplotlib.pyplot as plt

EXP_DIR = Path('experiments/2026-03-12_1835_codex_hardware-generalization-bakeoff')
RAW_DIR = EXP_DIR / 'analysis' / 'raw_logs'
MANIFEST = json.loads((EXP_DIR / 'manifest.json').read_text())
ANALYSIS = EXP_DIR / 'analysis'
ANALYSIS.mkdir(exist_ok=True)

manifest_by_app = {m['app_id']: m for m in MANIFEST}

PROBE_RULES = {
    'SLLQHLIGL': ('HLA-A*24:02', 'HLA-A*02:01'),
    'FLRYLLFGI': ('HLA-A*24:02', 'HLA-A*02:01'),
    'NFLIKFLLI': ('HLA-A*02:01', 'HLA-A*24:02'),
}

def safe_ratio(num, den):
    if den is None or den == 0 or num is None:
        return None
    return num / den

per_epoch = []
parsed = []

for log_path in sorted(RAW_DIR.glob('*.log')):
    app_id = log_path.stem
    meta = manifest_by_app[app_id]
    lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    setup = next(x for x in lines if x['event'] == 'focused_binding_setup')
    epochs = [x for x in lines if x['event'] == 'focused_binding_epoch']
    assert len(epochs) == 20, (log_path, len(epochs))

    # epoch rows
    for e in epochs:
        probe_map = {(r['peptide'], r['allele']): r for r in e['probe_rows']}
        row = {
            'app_id': app_id,
            'run_id': meta['run_id'],
            'requested_gpu': meta['requested_gpu'],
            'design_id': meta['design_id'],
            'epoch': e['epoch'],
            'train_loss': e['train_loss'],
            'val_loss': e['val_loss'],
            'epoch_wall_s': e['epoch_wall_s'],
            'train_data_wait_s': e['train_data_wait_s'],
            'train_forward_loss_s': e['train_forward_loss_s'],
            'train_backward_s': e['train_backward_s'],
            'train_optimizer_s': e['train_optimizer_s'],
            'gpu_util_mean_pct': e.get('gpu_util_mean_pct'),
            'gpu_mem_util_mean_pct': e.get('gpu_mem_util_mean_pct'),
            'gpu_peak_allocated_gib': e.get('gpu_peak_allocated_gib'),
            'gpu_peak_reserved_gib': e.get('gpu_peak_reserved_gib'),
            'current_lr': e.get('current_lr'),
        }
        for peptide, (num_allele, den_allele) in PROBE_RULES.items():
            num = probe_map.get((peptide, num_allele), {}).get('ic50_nM')
            den = probe_map.get((peptide, den_allele), {}).get('ic50_nM')
            row[f'{peptide}_num_nM'] = num
            row[f'{peptide}_den_nM'] = den
            row[f'{peptide}_ratio'] = safe_ratio(num, den)
            row[f'{peptide}_a0201_nM'] = probe_map.get((peptide, 'HLA-A*02:01'), {}).get('ic50_nM')
            row[f'{peptide}_a2402_nM'] = probe_map.get((peptide, 'HLA-A*24:02'), {}).get('ic50_nM')
        per_epoch.append(row)

    final = epochs[-1]
    best = min(epochs, key=lambda x: x['val_loss'])

    def get_probe(epoch, peptide, allele):
        for r in epoch['probe_rows']:
            if r['peptide'] == peptide and r['allele'] == allele:
                return r['ic50_nM']
        return None

    final_sll_a02 = get_probe(final, 'SLLQHLIGL', 'HLA-A*02:01')
    final_sll_a24 = get_probe(final, 'SLLQHLIGL', 'HLA-A*24:02')
    final_flr_a02 = get_probe(final, 'FLRYLLFGI', 'HLA-A*02:01')
    final_flr_a24 = get_probe(final, 'FLRYLLFGI', 'HLA-A*24:02')
    final_nfl_a02 = get_probe(final, 'NFLIKFLLI', 'HLA-A*02:01')
    final_nfl_a24 = get_probe(final, 'NFLIKFLLI', 'HLA-A*24:02')

    parsed.append({
        'app_id': app_id,
        'run_id': meta['run_id'],
        'requested_gpu': meta['requested_gpu'],
        'design_id': meta['design_id'],
        'lr': meta['lr'],
        'lr_schedule': meta['lr_schedule'],
        'batch_size': meta['batch_size'],
        'epochs': meta['epochs'],
        'setup_wall_s': setup['setup_wall_s'],
        'prepare_real_binding_state_s': setup.get('prepare_real_binding_state_s'),
        'dataset_build_s': setup.get('dataset_build_s'),
        'probe_setup_s': setup.get('probe_setup_s'),
        'mean_epoch_wall_s': mean(e['epoch_wall_s'] for e in epochs),
        'final_epoch_wall_s': final['epoch_wall_s'],
        'mean_train_data_wait_s': mean(e['train_data_wait_s'] for e in epochs),
        'mean_train_forward_loss_s': mean(e['train_forward_loss_s'] for e in epochs),
        'mean_train_backward_s': mean(e['train_backward_s'] for e in epochs),
        'mean_train_optimizer_s': mean(e['train_optimizer_s'] for e in epochs),
        'mean_gpu_util_pct': mean(e.get('gpu_util_mean_pct', 0) or 0 for e in epochs),
        'mean_gpu_mem_util_pct': mean(e.get('gpu_mem_util_mean_pct', 0) or 0 for e in epochs),
        'peak_allocated_gib': max((e.get('gpu_peak_allocated_gib', 0) or 0) for e in epochs),
        'peak_reserved_gib': max((e.get('gpu_peak_reserved_gib', 0) or 0) for e in epochs),
        'best_epoch': best['epoch'],
        'best_val_loss': best['val_loss'],
        'term_val_loss': final['val_loss'],
        'term_train_loss': final['train_loss'],
        'sll_a0201_nM': final_sll_a02,
        'sll_a2402_nM': final_sll_a24,
        'sll_ratio': safe_ratio(final_sll_a24, final_sll_a02),
        'flr_a0201_nM': final_flr_a02,
        'flr_a2402_nM': final_flr_a24,
        'flr_ratio': safe_ratio(final_flr_a24, final_flr_a02),
        'nfl_a0201_nM': final_nfl_a02,
        'nfl_a2402_nM': final_nfl_a24,
        'nfl_ratio': safe_ratio(final_nfl_a02, final_nfl_a24),
    })

# write structured data
per_epoch_csv = ANALYSIS / 'per_epoch_metrics.csv'
with per_epoch_csv.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(per_epoch[0].keys()))
    writer.writeheader()
    writer.writerows(per_epoch)
(ANALYSIS / 'per_epoch_metrics.json').write_text(json.dumps(per_epoch, indent=2))

parsed.sort(key=lambda x: (x['design_id'], x['requested_gpu']))
parsed_csv = ANALYSIS / 'parsed_metrics.csv'
with parsed_csv.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(parsed[0].keys()))
    writer.writeheader()
    writer.writerows(parsed)
(ANALYSIS / 'parsed_metrics.json').write_text(json.dumps(parsed, indent=2))

summary = {
    'winner_speed': min(parsed, key=lambda x: x['mean_epoch_wall_s'])['run_id'],
    'winner_a03_probe': max((r for r in parsed if r['design_id'].startswith('A03')), key=lambda x: (math.log10(x['sll_ratio']) + math.log10(x['flr_ratio']) + math.log10(x['nfl_ratio']))),
    'winner_a07_val': min((r for r in parsed if r['design_id'].startswith('A07')), key=lambda x: x['best_val_loss']),
}
(ANALYSIS / 'summary.json').write_text(json.dumps(summary, indent=2))

# plots
plt.style.use('default')
colors = {'A100':'#d55e00','H100!':'#0072b2','H200':'#009e73'}

def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

# per-design loss curves
for design_prefix in ['A03', 'A07']:
    plt.figure(figsize=(8,5))
    for gpu in ['A100','H100!','H200']:
        rows = [r for r in per_epoch if r['design_id'].startswith(design_prefix) and r['requested_gpu']==gpu]
        rows.sort(key=lambda x: x['epoch'])
        plt.plot([r['epoch'] for r in rows], [r['val_loss'] for r in rows], label=gpu, color=colors[gpu])
    plt.title(f'{design_prefix} val loss by epoch')
    plt.xlabel('epoch'); plt.ylabel('val loss'); plt.legend()
    save_plot(ANALYSIS / f'{design_prefix.lower()}_val_loss_by_epoch.png')

# per-design probe ratio curves (three subplots)
for design_prefix in ['A03', 'A07']:
    fig, axes = plt.subplots(1,3, figsize=(15,4))
    probe_specs = [
        ('SLLQHLIGL_ratio', 'SLL A24/A02'),
        ('FLRYLLFGI_ratio', 'FLR A24/A02'),
        ('NFLIKFLLI_ratio', 'NFL A02/A24'),
    ]
    for ax, (col, title) in zip(axes, probe_specs):
        for gpu in ['A100','H100!','H200']:
            rows = [r for r in per_epoch if r['design_id'].startswith(design_prefix) and r['requested_gpu']==gpu]
            rows.sort(key=lambda x: x['epoch'])
            ax.plot([r['epoch'] for r in rows], [r[col] for r in rows], label=gpu, color=colors[gpu])
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.set_ylabel('ratio')
        ax.set_yscale('log')
    axes[0].legend()
    fig.suptitle(f'{design_prefix} probe ratios by epoch')
    fig.tight_layout(rect=(0,0,1,0.95))
    fig.savefig(ANALYSIS / f'{design_prefix.lower()}_probe_ratios_by_epoch.png', dpi=180)
    plt.close(fig)

# combined speed vs best val scatter
plt.figure(figsize=(7,5))
for row in parsed:
    marker = 'o' if row['design_id'].startswith('A03') else 's'
    plt.scatter(row['mean_epoch_wall_s'], row['best_val_loss'], color=colors[row['requested_gpu']], marker=marker)
    plt.text(row['mean_epoch_wall_s'], row['best_val_loss'], f"{row['design_id'].split('_')[0]}-{row['requested_gpu'].replace('!','')}", fontsize=7)
plt.xlabel('mean epoch wall s')
plt.ylabel('best val loss')
plt.title('Hardware bakeoff: speed vs best val loss')
save_plot(ANALYSIS / 'speed_vs_best_val.png')

# combined final probe ratios per condition
fig, axes = plt.subplots(1,3, figsize=(15,4))
metrics = [('sll_ratio','SLL A24/A02'), ('flr_ratio','FLR A24/A02'), ('nfl_ratio','NFL A02/A24')]
labels = [f"{r['design_id'].split('_')[0]}-{r['requested_gpu'].replace('!','')}" for r in parsed]
for ax,(key,title) in zip(axes,metrics):
    vals=[r[key] for r in parsed]
    ax.bar(range(len(vals)), vals, color=[colors[r['requested_gpu']] for r in parsed])
    ax.set_title(title)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    ax.set_yscale('log')
fig.tight_layout()
fig.savefig(ANALYSIS / 'final_probe_ratios_by_condition.png', dpi=180)
plt.close(fig)

# markdown table
md_lines = []
md_lines.append('| design | gpu | best epoch | best val | term val | setup s | epoch s | gpu util % | peak alloc GiB | SLL A02 / A24 | SLL ratio | FLR A02 / A24 | FLR ratio | NFL A02 / A24 | NFL ratio |')
md_lines.append('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- | ---: |')
for r in parsed:
    md_lines.append(
        f"| `{r['design_id'].split('_')[0]}` | `{r['requested_gpu']}` | {r['best_epoch']} | {r['best_val_loss']:.4f} | {r['term_val_loss']:.4f} | {r['setup_wall_s']:.1f} | {r['mean_epoch_wall_s']:.1f} | {r['mean_gpu_util_pct']:.1f} | {r['peak_allocated_gib']:.2f} | {r['sll_a0201_nM']:.1f} / {r['sll_a2402_nM']:.1f} | {r['sll_ratio']:.1f} | {r['flr_a0201_nM']:.1f} / {r['flr_a2402_nM']:.1f} | {r['flr_ratio']:.1f} | {r['nfl_a0201_nM']:.1f} / {r['nfl_a2402_nM']:.1f} | {r['nfl_ratio']:.1f} |"
    )
(EXP_DIR / 'options_vs_perf.md').write_text('\n'.join(md_lines) + '\n')
(EXP_DIR / 'options_vs_perf.json').write_text(json.dumps(parsed, indent=2))

print('wrote', parsed_csv)
print('wrote', per_epoch_csv)
print('wrote', EXP_DIR / 'options_vs_perf.md')
