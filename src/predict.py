import os.path
import time
from mne.io import read_raw_edf
from tqdm import tqdm
from src.loader import edf_file, standardize_raw, get_epoch_and_labels
from sklearn.metrics import accuracy_score

from src.utils_pickle import get_data_from_pickle_file


def predict_bonus(process_dir, model):
    process_filename = os.path.join(process_dir, "test_data_bonus_dataset.pkl")
    (X_test, y_test) = get_data_from_pickle_file(process_filename, desc="Loading test data bonus dataset")
    y_pred = model.predict(X_test)
    print(f"Mean accuracy: {accuracy_score(y_test, y_pred):.3f}")


def eeg_bci_predict(data_dir, subjects, experiments, model, process_dir, bonus_dataset):
    if bonus_dataset:
        predict_bonus(process_dir, model)
        return
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
