import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extra_plot_and_save_confusion_matrices(results_dict,
                                     class_names_dict,
                                     output_dir="../results/figures"):
    """
    Plots and saves confusion matrices.

    Parameters
    ----------
    results_dict : dict
        same as before

    class_names_dict : dict
        {
            'control_vs_physio': ('Control', 'Physio'),
            ...
        }
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    aux = 3

    for name, (cm, f1, acc) in results_dict.items():

        class_names = class_names_dict[name]

        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

        fig, ax = plt.subplots(figsize=(5,4))

        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)

        # ticks
        ax.set_xticks([0,1])
        ax.set_yticks([0,1])
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        # labels
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{class_names[0]} vs {class_names[1]}")

        # values inside cells
        for i in range(2):
            for j in range(2):
                value = cm[i, j]
                color = "white" if cm_norm[i, j] > 0.5 else "black"

                ax.text(j, i, f"{value}",
                        ha="center", va="center",
                        color=color)

        fig.colorbar(im, ax=ax)

        # save
        filename = f"{aux:02d}_cm_{name}.png"
        filepath = output_dir / filename
        
        aux +=1

        plt.savefig(filepath, dpi=300, bbox_inches='tight')        
        plt.show()
        plt.close()

        print(f"Saved: {filepath}")