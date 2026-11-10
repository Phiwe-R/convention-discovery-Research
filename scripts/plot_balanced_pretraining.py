""" we are ignoring this file , its useless, i made plots that i didnt even use in my report

Plot genuine metrics from balanced supervised GRU pretraining.
   
Usage:
 python -m scripts.plot_balanced_pretraining --checkpoint results/joint_belief/balanced_exp2.pt
"""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def load_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return [{key: float(value) for key,value in row.items()}
                for row in csv.DictReader(f)]


def draw(checkpoint):
    prefix = Path(checkpoint).with_suffix('')
    history = load_csv(f'{prefix}_history.csv')
    timesteps = load_csv(f'{prefix}_test_by_timestep.csv')
    import torch
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    chance_c = 100 / state['num_conventions']
    chance_g = 100 / state['num_goals']
    outputs = []

    fig, ax = plt.subplots(figsize=(8, 4.7))
    x=[r['epoch'] for r in history]
    ax.plot(x, [r['val_convention_ce'] for r in history], marker='.', label='Convention CE')
    ax.plot(x, [r['val_goal_ce'] for r in history], marker='.', label='Goal CE')
    ax.plot(x, [r['val_movement_ce'] for r in history], marker='.', label='Next-action CE')
    ax.set(xlabel='Pretraining epoch', ylabel='Validation cross-entropy',
           title='Belief GRU: supervised validation losses')
    ax.legend()
    ax.grid(alpha=0.15)
    fig.tight_layout()
    dest=Path(f'{prefix}_validation_loss.png');fig.savefig(dest,dpi=220);plt.close(fig);outputs.append(dest)

    fig, ax = plt.subplots(figsize=(8, 4.7))
    ax.plot(x, [r['val_convention_final_pct'] for r in history], label='Convention ID')
    ax.plot(x, [r['val_goal_final_pct'] for r in history], label='Goal ID')
    ax.axhline(chance_c, linestyle='--', linewidth=1, label=f'Convention chance ({chance_c:.1f}%)')
    ax.axhline(chance_g, linestyle=':', linewidth=1, label=f'Goal chance ({chance_g:.1f}%)')
    ax.set(xlabel='Pretraining epoch', ylabel='Final identification accuracy (%)',
           title='Belief GRU: held-out validation identification')
    ax.legend()
    ax.grid(alpha=0.15)
    fig.tight_layout()
    dest=Path(f'{prefix}_validation_accuracy.png');fig.savefig(dest,dpi=220);plt.close(fig);outputs.append(dest)

    fig, ax = plt.subplots(figsize=(8, 4.7))
    x=[r['timestep'] for r in timesteps]
    ax.plot(x, [r['convention_accuracy_pct'] for r in timesteps], marker='o',label='Convention ID')
    ax.plot(x, [r['goal_accuracy_pct'] for r in timesteps], marker='o',label='Goal ID')
    ax.axhline(chance_c, linestyle='--', linewidth=1, label='Convention chance')
    ax.axhline(chance_g, linestyle=':', linewidth=1, label='Goal chance')
    ax.set(xlabel='Episode timestep', ylabel='Identification accuracy (%)',
           title='Belief GRU: independent test accuracy vs episode timestep')
    ax.set_xticks(range(1,13));ax.legend();ax.grid(alpha=0.15)
    fig.tight_layout()
    dest=Path(f'{prefix}_test_accuracy_by_timestep.png');fig.savefig(dest,dpi=220);plt.close(fig);outputs.append(dest)

    fig, ax = plt.subplots(figsize=(8, 4.7))
    ax.plot(x, [r['convention_true_probability'] for r in timesteps],marker='o',label='True convention probability')
    ax.plot(x, [r['goal_true_probability'] for r in timesteps],marker='o',label='True goal probability')
    ax.axhline(chance_c/100, linestyle='--',linewidth=1,label='Convention uniform probability')
    ax.axhline(chance_g/100, linestyle=':',linewidth=1,label='Goal uniform probability')
    ax.set(xlabel='Episode timestep',ylabel='Mean assigned probability',ylim=(0,1),
           title='Belief GRU: probability of correct hypothesis')
    ax.set_xticks(range(1,13));ax.legend();ax.grid(alpha=0.15)
    fig.tight_layout()
    dest=Path(f'{prefix}_test_true_probability.png');fig.savefig(dest,dpi=220);plt.close(fig);outputs.append(dest)
    for dest in outputs:print('Saved:',dest)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    draw(p.parse_args().checkpoint)

if __name__=='__main__':main()
