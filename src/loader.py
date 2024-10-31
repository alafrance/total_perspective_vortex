from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, pick_types
from mne.channels import make_standard_montage
import os
import pywt
import numpy as np
import mne
from tqdm import tqdm

mne.set_log_level("ERROR")


def edf_file(directory, subject, experiment):
    return os.path.join(directory, f'S{subject:03d}', f'S{subject:03d}R{experiment:02d}.edf')


def resample_raw(raw):
    target_sfreq = 160.0
    if raw.info['sfreq'] != target_sfreq:
        raw = raw.resample(target_sfreq)
    return raw


def load_data(data_dir, subjects, experiments):
    raw = concatenate_raws([
        resample_raw(read_raw_edf(edf_file(data_dir, subject, experiment), preload=True, verbose=False))
        for subject in tqdm(subjects, desc="Loading subject", dynamic_ncols=True, unit='M', unit_scale=True)
        for experiment in experiments
    ])
    print("Standardizing raw data...")
    raw = standardize_raw(raw)

    return get_epoch_and_labels(raw)


def apply_wavelet_transform(raw, wavelet='db4', level=4):
    """
    This function use the pywt library to apply wavelet transform by using 'ondelette' signal.
    The wavelet transform is more precise than fourier transform because the output frequence are localised with time

    Parameters
    ----------
    raw: mne.io.Raw
        An MNE Raw object containing the EEG data with annotations.
    wavelet: Wavelet object or name string, optional
        'db4' 'ondelette' family naming 'Daubechies' wuth 4 coefficient's filter. This 'ondelette' is often used in EEG analysis.
    level: int, optional
        level of decomposition (more is high, more is precise)
    Returns
    -------
    raw: mne.io.Raw
        An MNE Raw object containing the EEG data with annotations filtered by wavelet transform
    """
    data, times = raw.get_data(return_times=True)
    cleaned_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        coeffs = pywt.wavedec(data[i, :], wavelet, level=level)

        threshold = 0.04
        coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]

        cleaned_data[i, :] = pywt.waverec(coeffs, wavelet)

    raw._data = cleaned_data
    return raw


def standardize_raw(raw, display_plot=True):
    if display_plot:
        raw.plot()
    raw = apply_wavelet_transform(raw)
    if display_plot:
        raw.plot()
    eegbci.standardize(raw)
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)
    raw.set_eeg_reference(projection=True)
    raw.filter(16.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
    return raw


def get_epoch_and_labels(raw):
    event_id = {
        'T1': 1,
        'T2': 2
    }
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")
    epochs = Epochs(
        raw,
        event_id=event_id,
        tmin=0,
        tmax=2,
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
    )
    labels = epochs.events[:, -1] - 2
    return epochs, labels
