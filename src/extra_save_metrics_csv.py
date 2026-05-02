import numpy as np
import pandas as pd
from pathlib import Path

def extra_save_metrics_csv(results_dict, output_path="../results/metrics/metrics.csv"):
    """
    Saves metrics and confusion matrix components to CSV.

    Parameters
    ----------
    results_dict : dict
        Example:
        {
            'control_vs_physio': (cm_12, f1_12, acc_12),
            'control_vs_neuro':  (cm_13, f1_13, acc_13),
            'physio_vs_neuro':   (cm_23, f1_23, acc_23)
        }
    """

    rows = []

    for name, (cm, f1, acc) in results_dict.items():

        tn, fp = cm[0]
        fn, tp = cm[1]

        rows.append({
            'class_pair': name,
            'accuracy': round(acc, 4),
            'f1_score': round(f1, 4),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp)
        })

    df = pd.DataFrame(rows)

    # garantir diretório
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, sep=';', index=False)

    print(f"Metrics saved to: {output_path}")