# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import soundfile as sf
import os


def data_loader(dataset_name, data_root="../data", normalize=True):
    """
    Load metadata and audio signals from a dataset.

    Expected structure:
    data/<dataset_name>/<dataset_name>.csv
    data/<dataset_name>/<class>/<audio_file>

    CSV format (no header):
    [file, age, gender, class]

    Parameters
    ----------
    dataset_name : str
        Name of the dataset folder (e.g., "myUSP")

    data_root : str
        Root folder containing datasets (default: "../data")

    normalize : bool
        If True, normalize signals to max amplitude = 1

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with metadata + signal + class index

    fs_global : int or None
        Sampling frequency if consistent, else None

    class_map : dict
        Mapping from class names to integers
    """

    # -----------------------------
    # 1. Paths
    # -----------------------------
    dataset_path = os.path.join(data_root, dataset_name)
    csv_path = os.path.join(dataset_path, f"{dataset_name}.csv")

    # -----------------------------
    # 2. Load CSV
    # -----------------------------
    columns = ['file', 'age', 'gender', 'group']

    df = pd.read_csv(
        csv_path,
        delimiter=';',
        header=None,
        names=columns
    )

    # -----------------------------
    # 3. Build file paths
    # -----------------------------
    df['path'] = df.apply(
        lambda row: os.path.join(dataset_path, row['group'], row['file']),
        axis=1
    )

    # -----------------------------
    # 4. Encode group labels as integer class codes
    #    Note: encoding is alphabetical
    # -----------------------------
    df['class'] = pd.Categorical(df['group']).codes

    # -----------------------------
    # 5. Load signals
    # -----------------------------
    signals = []
    samplerates = []

    for _, row in df.iterrows():
        signal, fs = sf.read(row['path'])

        # Normalize (optional)
        if normalize and np.max(np.abs(signal)) > 0:
            signal = signal / np.max(np.abs(signal))

        signals.append(signal)
        samplerates.append(fs)

    df['signal'] = signals
    df['fs'] = samplerates

    # -----------------------------
    # 6. Check sampling rate
    # -----------------------------
    if df['fs'].nunique() == 1:
        fs_global = int(df['fs'].iloc[0])
        print(f"All signals have the same sampling rate: {fs_global} Hz")
    else:
        fs_global = None
        print("Warning: inconsistent sampling rates.")

    return df, fs_global