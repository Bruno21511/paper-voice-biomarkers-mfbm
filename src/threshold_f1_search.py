import numpy as np

def threshold_f1_search(X, y, step=0.1):
    """
    Finds threshold based on plateau of maximum F1.
    Returns midpoint of best F1 region.
    """

    std_values = X[:, 1]

    t_min = np.min(std_values)
    t_max = np.max(std_values)

    thresholds = np.arange(t_min, t_max, step)

    def compute_f1(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp + 1e-12)
        recall    = tp / (tp + fn + 1e-12)

        return 2 * precision * recall / (precision + recall + 1e-12)

    f1_values = []
    rules = []

    # --- compute F1 for all thresholds
    for t in thresholds:

        y_pred_1 = (std_values >= t).astype(int)
        f1_1 = compute_f1(y, y_pred_1)

        y_pred_2 = (std_values <= t).astype(int)
        f1_2 = compute_f1(y, y_pred_2)

        if f1_1 >= f1_2:
            f1_values.append(f1_1)
            rules.append("ge")
        else:
            f1_values.append(f1_2)
            rules.append("le")

    f1_values = np.array(f1_values)

    # --- find max F1
    best_f1 = np.max(f1_values)

    # --- find plateau (within tolerance)
    tol = 1e-3
    best_idxs = np.where(np.abs(f1_values - best_f1) < tol)[0]

    # --- get corresponding thresholds
    best_thresholds = thresholds[best_idxs]

    # --- midpoint of plateau
    t_opt = (best_thresholds[0] + best_thresholds[-1]) / 2

    return round(t_opt, 2)