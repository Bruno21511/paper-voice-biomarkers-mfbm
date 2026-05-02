import numpy as np
import matplotlib.pyplot as plt

def extra_plot_all_classes_std(
    df,
    std_band=2,
    thresholds=None,
    class_column='group',
    save_path=None
):

    # --- extrair banda correta
    std_values = df['std_MFBM'].apply(lambda x: x[std_band]).values
    classes = df[class_column].values


    x = np.arange(len(std_values))

    plt.figure(figsize=(8,4))

    unique_classes = np.unique(classes)
    markers = ['^', 'x', 'o']

    for i, c in enumerate(unique_classes):
        mask = classes == c
        plt.scatter(x[mask], std_values[mask],
                    label=c,
                    alpha=0.7,
                    marker=markers[i],
                    s = 40)

    # --- thresholds
    if thresholds:
        colors = ['tab:blue', 'tab:green', 'tab:green']
        linestyles = ['-.', '--', ':']

        for i, (name, t) in enumerate(thresholds.items()):
            plt.axhline(
                y=t,
                linestyle=linestyles[i % len(linestyles)],
                color=colors[i % len(colors)],
                linewidth=1.9,
                alpha=0.6,
                label=f'{name} threshold'
            )

    plt.xlabel("Instance index (sorted)")
    plt.ylabel(f"Std MFBM (band {std_band+1})")
    plt.title("Distribution of Std MFBM (Band 3) across all speakers by class")
    #plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()