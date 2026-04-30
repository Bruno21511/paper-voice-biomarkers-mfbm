# -*- coding: utf-8 -*-
import numpy as np
from itertools import combinations

def find_most_discriminative_bands(mean_dict, std_dict):
    
    results = []

    class_pairs = list(combinations(mean_dict.keys(), 2))

    for c1, c2 in class_pairs:

        # --- diferenças absolutas
        diff_mean = np.abs(mean_dict[c1] - mean_dict[c2])
        diff_std  = np.abs(std_dict[c1]  - std_dict[c2])

        # --- melhor banda
        best_band_mean = np.argmax(diff_mean)
        best_band_std  = np.argmax(diff_std)

        results.append({
            'pair': f"{c1} vs {c2}",
            'best_band_mean': best_band_mean,
            'diff_mean': diff_mean[best_band_mean],
            'best_band_std': best_band_std,
            'diff_std': diff_std[best_band_std]
        })
        
    print('Most discriminative frequency bands per class pair (Mean and Std criteria):')
        
    for r in results:
        print(f"\n{r['pair']}")
        print(f"  Mean → mel band {r['best_band_mean']+1}")
        print(f"  Std  → mel band {r['best_band_std']+1} ")

    return results