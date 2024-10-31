import time
from mne.io import read_raw_edf
from tqdm import tqdm
from src.loader import edf_file, standardize_raw, get_epoch_and_labels
from sklearn.metrics import accuracy_score


def eeg_bci_predict(data_dir, subjects, experiments, model):
    predictions = []
    true_labels = []
    for subject in tqdm(subjects, desc="Predict subjects", dynamic_ncols=True, unit='M', unit_scale=True):
        for experiment in experiments:
            raw = read_raw_edf(edf_file(data_dir, subject, experiment), preload=True, verbose=False)
            standardize_raw(raw, False)
            epochs, labels = get_epoch_and_labels(raw)

            start_time = time.time()
            y_pred = model.predict(epochs.get_data())
            delay = time.time() - start_time

            if delay > 2:
                print(f"Warning: Prediction took {delay:.2f} seconds")
            predictions.extend(y_pred)
            true_labels.extend(labels)

    print(f"Mean accuracy: {accuracy_score(true_labels, predictions):.3f}")
