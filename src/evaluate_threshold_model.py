import numpy as np
import matplotlib.pyplot as plt

def evaluate_threshold_model(X, y, threshold, class_names=('Class 0', 'Class 1'), save_path=None):
    """
    Evaluates threshold classifier on std feature (X[:,1]).

    Returns:
        confusion matrix (2x2)
        f1 score (rounded)
        accuracy (rounded)
    """

    std_values = X[:, 1]

    def compute_metrics(y, y_pred):
        tp = np.sum((y == 1) & (y_pred == 1))
        tn = np.sum((y == 0) & (y_pred == 0))
        fp = np.sum((y == 0) & (y_pred == 1))
        fn = np.sum((y == 1) & (y_pred == 0))

        precision = tp / (tp + fp + 1e-12)
        recall    = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-12)

        cm = np.array([[tn, fp],
                       [fn, tp]])

        return cm, f1, accuracy, tp, tn, fp, fn

    # --- first rule
    y_pred = (std_values >= threshold).astype(int)
    cm, f1, acc, tp, tn, fp, fn = compute_metrics(y, y_pred)

    # --- invert if needed
    if f1 < 0.5:
        y_pred = (std_values <= threshold).astype(int)
        cm, f1, acc, tp, tn, fp, fn = compute_metrics(y, y_pred)

    f1 = round(f1, 4)
    acc = round(acc, 4)

    # ======================================================
    # PLOT
    # ======================================================
    plt.figure(figsize=(6,5))

    plt.scatter(X[y == 0, 0], X[y == 0, 1],
                label=class_names[0], alpha=0.7, marker='o')

    plt.scatter(X[y == 1, 0], X[y == 1, 1],
                label=class_names[1], alpha=0.7, marker='x')

    plt.axhline(y=threshold, linestyle='--', color='black',
                label=f'Threshold = {threshold:.2f}')

    plt.xlabel("Mean MFBM")
    plt.ylabel("Std MFBM (band 3)")
    plt.title(f"{class_names[0]} vs {class_names[1]} — Accuracy: {acc:.2%}")
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

    # ======================================================
    # PRINT CONFUSION MATRIX
    # ======================================================
    print("Confusion Matrix:")
    print(f"{'':18s} Pred {class_names[0]:<8s}  Pred {class_names[1]:<8s}")
    print(f"True {class_names[0]:<8s} -> {tn:10d}   {fp:9d}")
    print(f"True {class_names[1]:<8s} -> {fn:10d}   {tp:9d}")

    print(f"\nF1-score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")

    return cm, f1, acc