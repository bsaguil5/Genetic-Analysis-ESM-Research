
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

import json

# Set publication-quality style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14

# Ensure directory exists
output_dir = "report_figures"
os.makedirs(output_dir, exist_ok=True)

# Path to the actual logs from the Victory Lap
TRAINING_LOG_PATH = "archive_legacy_2025/logs/results/training_log_20251220_194052.json"
TEST_RESULTS_PATH = "archive_legacy_2025/logs/results/test_results_20251220_194056.json"

def plot_dataset_composition():
    labels = ['Transporters (Label 0)', 'Non-Transporters (Label 1)']
    sizes = [795, 300]
    colors = ['#4CAF50', '#FF5722']
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 14})
    plt.title('Updated Dataset Composition (n=1,095)', fontsize=16)
    plt.savefig(f"{output_dir}/new_fig2_dataset_composition.png", dpi=300)
    print("✅ Generated Dataset Composition Chart")

def plot_confusion_matrix():
    # From Victory Lap Results:
    # Actual 0: 82 correct, 4 wrong
    # Actual 1: 129 correct, 4 wrong
    cm = np.array([[82, 4], [4, 129]])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 16})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix (Test Set n=219)', fontsize=14)
    plt.xticks([0.5, 1.5], ['Transporter', 'Non-Transporter'])
    plt.yticks([0.5, 1.5], ['Transporter', 'Non-Transporter'])
    plt.savefig(f"{output_dir}/new_fig4_confusion_matrix.png", dpi=300)
    print("✅ Generated Confusion Matrix")

def plot_benchmark():
    models = ['Random', 'Feedforward', 'BiGRU', 'Transformer', 'ESM-2 (Ours)']
    f1_scores = [0.50, 0.70, 0.85, 0.935, 0.97]
    colors = ['gray', 'gray', 'gray', 'gray', '#2196F3']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, f1_scores, color=colors)
    plt.ylim(0, 1.1)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Model Performance Benchmark', fontsize=14)
    
    # Add values on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=11)
                
    plt.savefig(f"{output_dir}/new_fig6_benchmark.png", dpi=300)
    print("✅ Generated Benchmark Chart")

def plot_training_curve():
    if not os.path.exists(TRAINING_LOG_PATH):
        print(f"⚠️ Could not find training log at {TRAINING_LOG_PATH}. Skipping.")
        return

    with open(TRAINING_LOG_PATH, 'r') as f:
        log_data = json.load(f)
    
    # Extract training loss from batch logs
    train_log = log_data.get('training_log', [])
    losses = [entry['loss'] for entry in train_log]
    
    # Calculate moving average for smoother curve
    window_size = 20
    if len(losses) > window_size:
        smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
    else:
        smoothed_losses = losses
        
    steps = range(len(smoothed_losses))
    
    plt.figure(figsize=(8, 6))
    plt.plot(steps, smoothed_losses, label='Training Loss', color='#2196F3', linewidth=2)
    plt.xlabel('Training Steps (Batches)')
    plt.ylabel('Focal Loss')
    plt.title('Training Convergence')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Add final metrics annotation if available
    if 'final_metrics' in log_data:
        final_f1 = log_data['final_metrics'].get('f1_score', 'N/A')
        final_acc = log_data['final_metrics'].get('accuracy', 'N/A')
        plt.annotate(f"Final F1: {final_f1:.4f}\nFinal Acc: {final_acc:.4f}", 
                     xy=(0.7, 0.8), xycoords='axes fraction', 
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    plt.savefig(f"{output_dir}/new_fig3_training_curve.png", dpi=300)
    print("✅ Generated Training Curve")

def plot_confidence_distribution():
    if not os.path.exists(TEST_RESULTS_PATH):
        print(f"⚠️ Could not find test results at {TEST_RESULTS_PATH}. Skipping.")
        return
        
    with open(TEST_RESULTS_PATH, 'r') as f:
        results = json.load(f)
        
    probs = [p for p in results['probabilities']]
    
    plt.figure(figsize=(8, 6))
    plt.hist(probs, bins=20, color='purple', alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Probability (0=Transporter, 1=Non-Transporter)')
    plt.ylabel('Count')
    plt.title('Prediction Confidence Distribution')
    plt.axvline(x=0.5, color='red', linestyle='--', label='Decision Boundary')
    plt.legend()
    
    plt.savefig(f"{output_dir}/new_fig8_confidence.png", dpi=300)
    print("✅ Generated Confidence Histogram")

def main():
    print("🎨 Generating Report Figures...")
    plot_dataset_composition()
    plot_confusion_matrix()
    plot_benchmark()
    plot_training_curve()
    plot_confidence_distribution()
    print("✨ All figures saved to report_figures/")

if __name__ == "__main__":
    main()
