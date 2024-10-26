from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
from src.loader.eeg_utils import load_specific_eeg, extract_epochs, EEGData
from src.loader.wavelet_transform import apply_wavelet_transform
from src.loader.utils import get_subject_name_by_id, get_experiment_name_by_id


def pipeline_loader_data(
        subjects,
        experiments,
        data_dir,
):
    """

    Parameters
    ----------
    subjects
    experiments
    data_dir

    Returns
    -------

    """
    eeg_data = load_eeg_data_with_signal_transform(
        data_dir=data_dir,
        subjects=subjects,
        experiments=experiments)
    X_train, X_val, X_test, y_train, y_val, y_test = spliter_train_test_val(eeg_data)

    data = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
    }
    return data


def load_eeg_data_with_signal_transform(
        data_dir,
        subjects,
        experiments
):
    """
    Load EEG data by subjects and experiments
    Parameters
    ----------
    data_dir: str
        Directory which can find all raw datas
    experiments: List[int]
        Array of experiments
    subjects: List[int]
        Range of subjects
    Returns
    -------
    data: list of EEGData
        All my datas information
    """
    data = []
    bar = tqdm(subjects, desc='Loading subject', dynamic_ncols=True, unit='M', unit_scale=True)
    plot_first_time = True
    for subject_id in bar:
        subject_name = get_subject_name_by_id(subject_id)
        bar.set_description(f"Loading subject {subject_name}")
        for experiment_id in experiments:
            experiment_name = get_experiment_name_by_id(experiment_id)

            # EEG info
            raw = load_specific_eeg(data_dir, subject_name, experiment_name)

            if plot_first_time:
                raw.plot(scalings='auto', title='Raw EEG data', verbose=False)
            # Preprocessing signal transformation
            raw = apply_wavelet_transform(raw)
            if plot_first_time:
                raw.plot(scalings='auto', title='Raw EEG data filtered', verbose=False)
                plot_first_time = False

            # Extract epochs from filtered raw
            epoch_data, epoch_labels = extract_epochs(raw)

            # Append my data
            data.append(EEGData(
                epoch_data=epoch_data,
                epoch_labels=epoch_labels,
                subject_id=subject_id,
                experiment_id=experiment_id
            ))
    return data


def spliter_train_test_val(eeg_data: EEGData):
    max_dim = max(eegdata.epoch_data.shape[1] for eegdata in eeg_data)
    first_epoch = True

    for epoch_data, epoch_labels, subject_id, experiment_id in eeg_data:
        epoch_data = padding_data(epoch_data, max_dim)
        X_train, X_val, X_test, y_train, y_val, y_test = train_test_val_split(
            epoch_data, epoch_labels, stratify=epoch_labels
        )
        if first_epoch:
            X_train_all = X_train
            X_val_all = X_val
            X_test_all = X_test
            y_train_all = y_train
            y_val_all = y_val
            y_test_all = y_test
            first_epoch = False
        else:
            X_train_all = np.concatenate((X_train_all, X_train), axis=0)
            X_val_all = np.concatenate((X_val_all, X_val), axis=0)
            X_test_all = np.concatenate((X_test_all, X_test), axis=0)
            y_train_all = np.concatenate((y_train_all, y_train), axis=0)
            y_val_all = np.concatenate((y_val_all, y_val), axis=0)
            y_test_all = np.concatenate((y_test_all, y_test), axis=0)

    return X_train_all, X_val_all, X_test_all, y_train_all, y_val_all, y_test_all



def padding_data(data, max_dim):
    n_feature = data.shape[1]

    if n_feature < max_dim:
        pad_width = ((0, 0), (0, max_dim - n_feature))
        data = np.pad(data, pad_width, mode='constant', constant_values=0)
    return data


def train_test_val_split(X, y, test_size=0.2, val_size=0.25, random_state=42, stratify=None):
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    val_size_adjusted = val_size / (1 - test_size)  # Adjust validation size proportionally
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=random_state, stratify=y_train_val
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
