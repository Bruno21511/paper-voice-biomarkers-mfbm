# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def plot_band_discriminability(mean_dict, c1, c2):
    """
    Plot band-wise discriminability between two classes.

    Parameters
    ----------
    mean_dict : dict
        Dictionary with class-wise mean MFBM (20 values per class)
    c1, c2 : str
        Class names to compare
    metric : str
        "abs" (default) → |a - b|
        "log"           → |log(a) - log(b)|
        "ratio"         → |a / (b + eps)|
    """

    eps = 1e-12

    a = mean_dict[c1]
    b = mean_dict[c2]

    # --- metric selection
    diff = np.abs(a - b)



    bandas = np.arange(len(diff))+1
    best = np.argmax(diff)

    # --- plot
    plt.figure(figsize=(10,5))
    plt.plot(bandas, diff, marker='o', label=f'Best band: {best + 1}')
    #plt.axvline(best+1, linestyle='--', label=f'Best band: {best + 1}')

    plt.title(f"{c1} vs {c2}")
    plt.xlabel("Band")
    plt.ylabel("Discriminability")
    plt.grid(True)
    plt.xticks(range(len(bandas)+1))
    plt.legend()
    plt.show()

    return