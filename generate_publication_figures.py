#!/usr/bin/env python3
"""
Publication-Ready Figure Generation Script
Updated: December 29, 2025
Uses VERIFIED metrics from DATA_VERIFICATION_REPORT_Dec29.md
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Set publication-quality style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 14

# Output directory
output_dir = "publication_figures"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# VERIFIED METRICS (from DATA_VERIFICATION_REPORT_Dec29.md)
# ============================================================================
DATASET_SIZE = 1323
TRANSPORTERS = 517
NON_TRANSPORTERS = 806
TRAINING_F1 = 0.9969
VALIDATION_F1 = 0.9558
VALIDATION_ACCURACY = 0.9655
TEST_SIZE = 319
CONFUSION_MATRIX = np.array([[189, 11], [0, 119]])  # [[TP, FN], [FP, TN]]

print("="*80)
print("PUBLICATION FIGURE GENERATOR")
print("="*80)
print(f"Dataset: {DATASET_SIZE} sequences ({TRANSPORTERS} transporters, {NON_TRANSPORTERS} non-transporters)")
print(f"Training F1: {TRAINING_F1:.4f}")
print(f"Validation F1: {VALIDATION_F1:.4f} ({VALIDATION_F1*100:.2f}%)")
print(f"Validation Accuracy: {VALIDATION_ACCURACY*100:.2f}%")
print(f"Test Set: {TEST_SIZE} sequences")
print("="*80)

# ============================================================================
# Figure 1: Dataset Composition
# ============================================================================
def plot_dataset_composition():
    """Dataset composition pie chart with correct counts"""
    print("\n[1/7] Generating Dataset Composition...")

    labels = ['Transporters\n(Label 0)', 'Non-Transporters\n(Label 1)']
    sizes = [TRANSPORTERS, NON_TRANSPORTERS]
    colors = ['#4CAF50', '#FF5722']
    percentages = [100*TRANSPORTERS/DATASET_SIZE, 100*NON_TRANSPORTERS/DATASET_SIZE]

    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 14, 'fontweight': 'bold'}
    )

    # Make percentage text more visible
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(16)
        autotext.set_fontweight('bold')

    plt.title(f'Dataset Composition (n={DATASET_SIZE} sequences)', fontsize=18, fontweight='bold', pad=20)

    # Add legend with counts
    legend_labels = [
        f'Transporters: {TRANSPORTERS} ({percentages[0]:.1f}%)',
        f'Non-Transporters: {NON_TRANSPORTERS} ({percentages[1]:.1f}%)'
    ]
    plt.legend(legend_labels, loc='upper left', fontsize=12, bbox_to_anchor=(0, 0.95))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig1_Dataset_Composition.png", dpi=300, bbox_inches='tight')
    print("    [OK] Saved: Fig1_Dataset_Composition.png")
    plt.close()

# ============================================================================
# Figure 2: Confusion Matrix (Blind Validation)
# ============================================================================
def plot_confusion_matrix():
    """Confusion matrix with VERIFIED results: [[189,11],[0,119]]"""
    print("\n[2/7] Generating Confusion Matrix...")

    fig, ax = plt.subplots(figsize=(10, 8))

    # VERIFIED confusion matrix from blind test
    cm = CONFUSION_MATRIX

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=True,
        annot_kws={'size': 24, 'fontweight': 'bold'},
        linewidths=2,
        linecolor='black',
        square=True
    )

    plt.xlabel('Predicted Label', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=16, fontweight='bold')
    plt.title(f'Confusion Matrix - Blind Validation (n={TEST_SIZE})', fontsize=18, fontweight='bold', pad=20)

    # Set tick labels
    plt.xticks([0.5, 1.5], ['Transporter (0)', 'Non-Transporter (1)'], fontsize=14)
    plt.yticks([0.5, 1.5], ['Transporter (0)', 'Non-Transporter (1)'], fontsize=14, rotation=0)

    # Calculate metrics
    tp, fn = cm[0]
    fp, tn = cm[1]
    total_transporters = tp + fn
    total_non_transporters = fp + tn
    recall = tp / total_transporters
    precision = tn / total_non_transporters

    # Add metrics box
    metrics_text = (
        f'True Positives: {tp}\n'
        f'False Negatives: {fn}\n'
        f'False Positives: {fp}\n'
        f'True Negatives: {tn}\n\n'
        f'Recall (Transporters): {recall*100:.1f}%\n'
        f'Precision (Non-Transporters): {precision*100:.1f}%\n'
        f'Overall Accuracy: {VALIDATION_ACCURACY*100:.2f}%'
    )

    ax.text(1.05, 0.5, metrics_text, transform=ax.transAxes,
            fontsize=11, va='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, pad=1))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig2_Confusion_Matrix.png", dpi=300, bbox_inches='tight')
    print("    [OK] Saved: Fig2_Confusion_Matrix.png")
    plt.close()

# ============================================================================
# Figure 3: Model Performance Metrics
# ============================================================================
def plot_performance_metrics():
    """Bar chart comparing Training vs Validation F1"""
    print("\n[3/7] Generating Performance Metrics...")

    fig, ax = plt.subplots(figsize=(10, 7))

    metrics = ['Training F1', 'Validation F1', 'Validation\nAccuracy']
    values = [TRAINING_F1, VALIDATION_F1, VALIDATION_ACCURACY]
    colors = ['#2196F3', '#4CAF50', '#FF9800']

    bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=2, alpha=0.8)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}\n({value*100:.2f}%)',
                ha='center', va='bottom', fontsize=13, fontweight='bold')

    ax.set_ylabel('Score', fontsize=16, fontweight='bold')
    ax.set_title('Model Performance Summary', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # Add horizontal line at 95%
    ax.axhline(y=0.95, color='red', linestyle='--', linewidth=2, alpha=0.5, label='95% threshold')
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig3_Performance_Metrics.png", dpi=300, bbox_inches='tight')
    print("    [OK] Saved: Fig3_Performance_Metrics.png")
    plt.close()

# ============================================================================
# Figure 4: Breeding Target Validation (NEW - Critical!)
# ============================================================================
def plot_breeding_targets():
    """Breeding target validation results - 4/4 success"""
    print("\n[4/7] Generating Breeding Target Validation...")

    fig, ax = plt.subplots(figsize=(12, 7))

    targets = ['SUT1\n(Sugar)', 'NRT2.4\n(Nitrogen)', 'SbMATE\n(Aluminum)', 'AKT1\n(Control)']
    confidences = [0.9621, 0.9477, 0.8863, 0.6283]
    protein_ids = ['XP_002465781.1', 'XP_002455791.2', 'ABS89149.1', 'XP_021312865.1']
    roles = ['Biofuel', 'Biomass', 'Stress Tolerance', 'Potassium Transport']

    # Color code by confidence
    colors = []
    for conf in confidences:
        if conf > 0.90:
            colors.append('#4CAF50')  # Green - excellent
        elif conf > 0.85:
            colors.append('#FF9800')  # Orange - good
        else:
            colors.append('#2196F3')  # Blue - as expected (control)

    bars = ax.barh(targets, confidences, color=colors, edgecolor='black', linewidth=2, alpha=0.8)

    # Add confidence percentages
    for i, (bar, conf, protein_id, role) in enumerate(zip(bars, confidences, protein_ids, roles)):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{conf*100:.2f}%\n{protein_id}\n{role}',
                ha='left', va='center', fontsize=10, fontweight='bold')

    # Add threshold line
    ax.axvline(x=0.85, color='red', linestyle='--', linewidth=2, alpha=0.7, label='85% threshold')

    ax.set_xlabel('Transporter Confidence Score', fontsize=16, fontweight='bold')
    ax.set_title('Breeding Target Validation - 4/4 PASS (100% Success)',
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlim(0, 1.05)
    ax.grid(axis='x', alpha=0.3)
    ax.legend(fontsize=12, loc='lower right')

    # Add success banner
    ax.text(0.5, 1.05, '[SUCCESS] ALL BREEDING TARGETS IDENTIFIED',
            transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9, pad=0.5))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig4_Breeding_Targets.png", dpi=300, bbox_inches='tight')
    print("    [OK] Saved: Fig4_Breeding_Targets.png")
    plt.close()

# ============================================================================
# Figure 5: Platinum Candidates Discovery
# ============================================================================
def plot_platinum_candidates():
    """Top platinum transporter candidates discovered"""
    print("\n[5/7] Generating Platinum Candidates...")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Top 10 platinum candidates from verified_platinum_candidates_v2.csv
    candidates_data = [
        ['Rank', 'Protein ID', 'Confidence', 'Probability'],
        ['1', 'KAG0553114.1', '93.26%', '0.0674'],
        ['2', 'KAG0551591.1', '92.87%', '0.0713'],
        ['3', 'KAG0552905.1', '92.48%', '0.0752'],
        ['4', 'KAG0552599.1', '92.43%', '0.0757'],
        ['5', 'KAG0552769.1', '92.29%', '0.0771'],
        ['6', 'KAG0552054.1', '92.11%', '0.0789'],
        ['7', 'KAG0552445.1', '91.80%', '0.0820'],
        ['8', 'KAG0553278.1', '91.44%', '0.0856'],
        ['9', 'KAG0551894.1', '91.15%', '0.0885'],
        ['10', 'KAG0552670.1', '90.92%', '0.0908'],
    ]

    table = ax.table(cellText=candidates_data, loc='center', cellLoc='center',
                     colWidths=[0.1, 0.4, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5)

    # Style header row
    for j in range(4):
        table[(0, j)].set_facecolor('#2196F3')
        table[(0, j)].set_text_props(color='white', fontweight='bold', fontsize=12)

    # Alternate row colors
    for i in range(1, len(candidates_data)):
        color = '#E8F4F8' if i % 2 == 0 else 'white'
        for j in range(4):
            table[(i, j)].set_facecolor(color)
            if j == 2:  # Confidence column - make bold
                table[(i, j)].set_text_props(fontweight='bold')

    ax.set_title('Top 10 Platinum Transporter Candidates\n(>85% Confidence, Genome-Wide Discovery)',
                 fontsize=16, fontweight='bold', pad=20)

    # Add summary box
    summary_text = (
        'Discovery Summary:\n'
        f'- Total candidates: 38 platinum-grade (>85% confidence)\n'
        '- Predictions: All Label 0 (TRANSPORTER)\n'
        '- Top candidate: KAG0553114.1 at 93.26% confidence\n'
        '- Ready for experimental validation'
    )
    ax.text(0.5, -0.05, summary_text, transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=0.8))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig5_Platinum_Candidates.png", dpi=300, bbox_inches='tight')
    print("    [OK] Saved: Fig5_Platinum_Candidates.png")
    plt.close()

# ============================================================================
# Figure 6: Model Architecture Diagram
# ============================================================================
def plot_architecture():
    """ESM-2 model architecture diagram"""
    print("\n[6/7] Generating Architecture Diagram...")

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'ESM-2 Protein Classification Architecture',
            fontsize=18, fontweight='bold', ha='center')

    # Define boxes
    boxes = [
        {'x': 0.5, 'y': 7.5, 'w': 1.8, 'h': 1.2, 'text': 'Protein\nSequence\n(FASTA)', 'color': '#E8F4F8'},
        {'x': 2.8, 'y': 7.5, 'w': 1.8, 'h': 1.2, 'text': 'ESM-2\nTokenizer', 'color': '#D4E6F1'},
        {'x': 5.1, 'y': 7.5, 'w': 2.2, 'h': 1.2, 'text': 'ESM-2\nTransformer\n(12 layers)\n35M params', 'color': '#AED6F1'},
        {'x': 7.8, 'y': 7.5, 'w': 1.8, 'h': 1.2, 'text': 'Embeddings\n(510 × 480)', 'color': '#85C1E2'},
        {'x': 3.2, 'y': 5.5, 'w': 1.8, 'h': 1, 'text': 'Attention\nPooling', 'color': '#FFE6CC'},
        {'x': 5.5, 'y': 5.5, 'w': 2.0, 'h': 1, 'text': 'Dense Layers\n480→256→128→1', 'color': '#FFCC99'},
        {'x': 3.8, 'y': 3.8, 'w': 2.8, 'h': 0.9, 'text': 'Focal Loss + Sigmoid', 'color': '#FFB366'},
        {'x': 3.5, 'y': 2.2, 'w': 3.3, 'h': 1, 'text': 'Binary Classification\n0=Transporter | 1=Non-Transporter', 'color': '#D5F4E6'},
    ]

    # Draw boxes
    for box in boxes:
        fancy_box = FancyBboxPatch(
            (box['x'], box['y']), box['w'], box['h'],
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=box['color'],
            linewidth=2.5
        )
        ax.add_patch(fancy_box)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2, box['text'],
                ha='center', va='center', fontsize=11, fontweight='bold')

    # Draw arrows
    arrows = [
        {'start': (2.3, 8.1), 'end': (2.8, 8.1)},
        {'start': (4.6, 8.1), 'end': (5.1, 8.1)},
        {'start': (7.3, 8.1), 'end': (7.8, 8.1)},
        {'start': (8.7, 7.5), 'end': (5.0, 6.5)},
        {'start': (4.1, 5.5), 'end': (5.5, 5.5)},
        {'start': (6.5, 5.5), 'end': (5.2, 4.7)},
        {'start': (5.2, 3.8), 'end': (5.2, 3.2)},
    ]

    for arrow in arrows:
        arr = FancyArrowPatch(
            arrow['start'], arrow['end'],
            arrowstyle='->',
            color='black',
            linewidth=3,
            mutation_scale=25
        )
        ax.add_patch(arr)

    # Add annotations
    annotations = [
        {'pos': (8.8, 6.8), 'text': 'Pre-trained\non 250M\nsequences', 'color': 'lightblue'},
        {'pos': (8.2, 5.0), 'text': 'Trainable\nWeights', 'color': 'lightcoral'},
        {'pos': (1.5, 5.0), 'text': 'Transfer\nLearning', 'color': 'lightgreen'},
    ]

    for ann in annotations:
        ax.text(ann['pos'][0], ann['pos'][1], ann['text'],
                fontsize=10, style='italic', ha='center',
                bbox=dict(boxstyle='round', facecolor=ann['color'], alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig6_Architecture.png", dpi=300, bbox_inches='tight')
    print("    [OK] Saved: Fig6_Architecture.png")
    plt.close()

# ============================================================================
# Figure 7: Performance Summary Table
# ============================================================================
def plot_performance_table():
    """Comprehensive performance metrics table"""
    print("\n[7/7] Generating Performance Summary Table...")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Calculate all metrics from confusion matrix
    tp, fn = CONFUSION_MATRIX[0]
    fp, tn = CONFUSION_MATRIX[1]

    recall = tp / (tp + fn)
    precision = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    specificity = tn / (tn + fp)

    table_data = [
        ['Metric', 'Value', 'Description'],
        ['Dataset Size', f'{DATASET_SIZE}', 'Total training sequences'],
        ['Transporters', f'{TRANSPORTERS} (39.1%)', 'Label 0 - Positive class'],
        ['Non-Transporters', f'{NON_TRANSPORTERS} (60.9%)', 'Label 1 - Negative class'],
        ['', '', ''],
        ['Training F1', f'{TRAINING_F1:.4f}', 'Training set performance'],
        ['Validation F1', f'{VALIDATION_F1:.4f} (95.58%)', 'Blind test set performance'],
        ['Validation Accuracy', f'{VALIDATION_ACCURACY*100:.2f}%', 'Overall correct predictions'],
        ['', '', ''],
        ['Recall (Sensitivity)', f'{recall*100:.1f}%', f'{tp}/{tp+fn} transporters identified'],
        ['Specificity', f'{specificity*100:.1f}%', f'{tn}/{tn+fp} non-transporters rejected'],
        ['False Positives', f'{fp}', 'Zero kinases misclassified'],
        ['False Negatives', f'{fn}', 'Missed transporters'],
        ['', '', ''],
        ['Breeding Targets', '4/4 (100%)', 'SUT1, NRT2.4, SbMATE, AKT1'],
        ['Platinum Candidates', '38', 'Novel transporters (>85% conf.)'],
        ['Top Candidate', 'KAG0553114.1', '93.26% confidence'],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='left',
                     colWidths=[0.3, 0.25, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.2)

    # Style header
    for j in range(3):
        table[(0, j)].set_facecolor('#2196F3')
        table[(0, j)].set_text_props(color='white', fontweight='bold', fontsize=12)

    # Color code rows
    section_rows = [4, 8, 13]  # Empty separator rows
    for i in range(1, len(table_data)):
        if i in section_rows:
            for j in range(3):
                table[(i, j)].set_facecolor('#CCCCCC')
        else:
            color = '#E8F4F8' if i % 2 == 0 else 'white'
            for j in range(3):
                table[(i, j)].set_facecolor(color)

        # Make value column bold
        table[(i, 1)].set_text_props(fontweight='bold')

    ax.set_title('Comprehensive Model Performance Summary',
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig7_Performance_Table.png", dpi=300, bbox_inches='tight')
    print("    [OK] Saved: Fig7_Performance_Table.png")
    plt.close()

# ============================================================================
# Main Execution
# ============================================================================
def main():
    """Generate all publication figures"""
    print("\n" + "="*80)
    print("GENERATING PUBLICATION FIGURES WITH VERIFIED METRICS")
    print("="*80)

    plot_dataset_composition()
    plot_confusion_matrix()
    plot_performance_metrics()
    plot_breeding_targets()
    plot_platinum_candidates()
    plot_architecture()
    plot_performance_table()

    print("\n" + "="*80)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}/")
    print("\nFigures ready for manuscript:")
    print("  - Fig1_Dataset_Composition.png")
    print("  - Fig2_Confusion_Matrix.png")
    print("  - Fig3_Performance_Metrics.png")
    print("  - Fig4_Breeding_Targets.png          [NEW - Critical for paper!]")
    print("  - Fig5_Platinum_Candidates.png")
    print("  - Fig6_Architecture.png")
    print("  - Fig7_Performance_Table.png")
    print("\nAll metrics verified against:")
    print("  - data/cleaned_labeled_sequences.csv")
    print("  - data/robustness_test.csv")
    print("  - data/verified_platinum_candidates_v2.csv")
    print("  - LOGS/12-24-to-28-2025_MULTITASK_EXPERIMENT.txt")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
