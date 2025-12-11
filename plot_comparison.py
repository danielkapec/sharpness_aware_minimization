import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def plot_runs(baseline_path, sam_path):
    # 1. Load Data
    try:
        df_base = pd.read_csv(baseline_path)
        df_sam = pd.read_csv(sam_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find file. Check your paths!\n{e}")
        return
    

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Baseline (SGD) vs. SAM - CIFAR-10 Training', fontsize=16)

    base_style = {'color': 'tab:blue', 'linestyle': '--', 'label': 'Baseline (SGD)'}
    sam_style = {'color': 'tab:orange', 'linestyle': '-', 'label': 'SAM'}

    ax = axes[0, 0]
    ax.plot(df_base['epoch'], df_base['train_loss'], **base_style)
    ax.plot(df_sam['epoch'], df_sam['train_loss'], **sam_style)
    ax.set_title('Training Loss (Lower is Better)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(df_base['epoch'], df_base['test_loss'], **base_style)
    ax.plot(df_sam['epoch'], df_sam['test_loss'], **sam_style)
    ax.set_title('Test Loss (Lower is Better)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(df_base['epoch'], df_base['train_acc'], **base_style)
    ax.plot(df_sam['epoch'], df_sam['train_acc'], **sam_style)
    ax.set_title('Training Accuracy (Higher is Better)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.grid(True, alpha=0.3)
    ax.legend()


    ax = axes[1, 1]
    ax.plot(df_base['epoch'], df_base['test_acc'], **base_style)
    ax.plot(df_sam['epoch'], df_sam['test_acc'], **sam_style)
    ax.set_title('Test Accuracy (Higher is Better)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.grid(True, alpha=0.3)
    ax.legend()

    best_base = df_base['test_acc'].max()
    best_sam = df_sam['test_acc'].max()
    
    print(f"Results Summary:")
    print(f"Baseline Best Test Acc: {best_base:.4f}")
    print(f"SAM Best Test Acc:      {best_sam:.4f}")
    print(f"Difference:             {(best_sam - best_base)*100:.2f}%")

    # Layout adjustment
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    BASELINE_CSV = "experiments/run_20251210_102211/log.csv" 
    
    # Example: "experiments_sam/run_20231025_120000/log.csv"
    SAM_CSV = "experiments_sam/run_20251210_133426/log.csv"

    plot_runs(BASELINE_CSV, SAM_CSV)