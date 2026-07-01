"""
Assignment #2 - Visualization
1. Confusion Matrix Heatmap (V5)
2. Macro F1 Comparison: V2 vs V5
3. Per-Class F1 Comparison: V2 vs V5 (Task 2)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. Confusion Matrix Heatmap (V5)
# ============================================================
cm_task1 = np.array([[203, 38],
                     [ 26, 133]])

cm_task2 = np.array([[203, 16,  8, 14],
                     [ 17, 40,  7, 11],
                     [  3,  8, 20,  4],
                     [  6,  8,  2, 33]])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm_task1, annot=True, fmt='d', cmap='Blues',
            xticklabels=["T0", "T2-T4"],
            yticklabels=["T0", "T2-T4"],
            ax=axes[0])
axes[0].set_title("Task 1: Binary Classification\nConfusion Matrix", fontsize=13)
axes[0].set_xlabel("Predicted", fontsize=11)
axes[0].set_ylabel("True", fontsize=11)

sns.heatmap(cm_task2, annot=True, fmt='d', cmap='Blues',
            xticklabels=["T0", "T2", "T3", "T4"],
            yticklabels=["T0", "T2", "T3", "T4"],
            ax=axes[1])
axes[1].set_title("Task 2: Multi-Class Classification\nConfusion Matrix", fontsize=13)
axes[1].set_xlabel("Predicted", fontsize=11)
axes[1].set_ylabel("True", fontsize=11)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("已儲存: confusion_matrix.png")

# ============================================================
# 2. Macro F1 Comparison: V2 vs V5
# ============================================================
versions      = ["V2\n(CrossEntropy)",
                 "V5\n(Focal Loss)"]
task1_f1      = [0.8261, 0.8349]
task2_f1      = [0.6206, 0.6395]

x     = np.arange(len(versions))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))
bars1 = ax.bar(x - width/2, task1_f1, width, label='Task 1', color='steelblue', alpha=0.85)
bars2 = ax.bar(x + width/2, task2_f1, width, label='Task 2', color='coral',     alpha=0.85)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Macro F1', fontsize=12)
ax.set_title('Macro F1 Comparison: V2 vs V5', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(versions, fontsize=10)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("macro_f1_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("已儲存: macro_f1_comparison.png")

# ============================================================
# 3. Per-Class F1 Comparison: V2 vs V5 (Task 2)
# ============================================================
classes = ["T0", "T2",
           "T3", "T4"]

v2_f1 = [0.8489, 0.5278, 0.5474, 0.5586]
v5_f1 = [0.8638, 0.5442, 0.5556, 0.5946]

x     = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
bars_v2 = ax.bar(x - width/2, v2_f1, width,
                 label='CrossEntropyLoss',
                 color='steelblue', alpha=0.85)
bars_v5 = ax.bar(x + width/2, v5_f1, width,
                 label='Focal Loss',
                 color='coral', alpha=0.85)

for bar in bars_v2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars_v5:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Task 2 Per-Class F1: CrossEntropyLoss vs Focal Loss', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=10)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("per_class_f1_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("已儲存: per_class_f1_comparison.png")

# ============================================================
# 4. Per-Class F1 Comparison: V2 vs V5 (Task 1)
# ============================================================
classes_t1 = ["T0\n(bench/basic)", "T2/T4\n(translational)"]
v2_f1_t1   = [0.8595, 0.7926]
v5_f1_t1   = [0.8638, 0.8061]

x     = np.arange(len(classes_t1))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))
bars_v2 = ax.bar(x - width/2, v2_f1_t1, width,
                 label='V2 (CrossEntropy + Class Weights, Single-Stage)',
                 color='steelblue', alpha=0.85)
bars_v5 = ax.bar(x + width/2, v5_f1_t1, width,
                 label='V5 (Focal Loss + Class Weights, Two-Stage)',
                 color='coral', alpha=0.85)

for bar in bars_v2:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
for bar in bars_v5:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)

ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Task 1 Per-Class F1: V2 vs V5', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(classes_t1, fontsize=11)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("per_class_f1_task1_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("已儲存: per_class_f1_task1_comparison.png")

print("\n全部圖表已產生！")
