import os

import mne


def load_specific_eeg(directory, subject, experiment, l_freq=8, h_freq=40):
    """

    Parameters
    ----------
    directory: str
        Directory of eeg file
    subject: str
        Subject name
    experiment: str
        Experiment name
    l_freq: int
        Low frequency to filter lower freq
    h_freq: int
        High frequency to filter higher freq

    Returns
    -------
    raw: mne.io.Raw
        An MNE Raw object containing the EEG data with annotations.
    filtered_raw: mne.io.Raw
        Filtered raw
    """
    subject_dir = os.path.join(directory, subject)
    edf_file = os.path.join(subject_dir, f'{subject}{experiment}.edf')

    # Load data
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)

    # FIR (Finite Impulse Response) : Filter method
    fir_design = 'firwin'

    # Remove undesirable raw data from low and high frequency
    filtered_raw = raw.copy().filter(l_freq, h_freq, fir_design=fir_design, verbose=False)

    return raw, filtered_raw


def extract_epochs(raw, tmin=0, tmax=1):
    """
    Extract epoch data and labels from raw data file
    Parameters
    ----------
    tmin : int
     Minimum time in seconds to get data and create epoch
    tmax : int
     Maximym time in seconds to get data and create epoch
    raw : mne.io.Raw
        An MNE Raw object containing the EEG data with annotations.
    Returns
    -------
    epoch_data : array of shape (n_epochs, n_channels, n_times)
        Epoch data
    epoch_labels : list
        Epoch labels
    """
    event_id = {
        'T0': 0,
        'T1': 1,
        'T2': 2
    }
    # Get events from annotations T0, T1, T2
    events, event_dict = mne.events_from_annotations(raw, event_id=event_id, verbose=False)

    # Remove labels from events for epochs
    labels = events[:, -1]
    mask = labels != 0
    events = events[mask]

    # Pick only eeg Canal
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

    # Create epochs from events from annotations
    epochs = mne.Epochs(raw, events, event_id=None, tmin=tmin, tmax=tmax, proj=True, picks=picks, baseline=None, preload=True, verbose=False)

    if len(epochs) == 0:
        raise ValueError('No epochs extracted')
    # Get data epochs
    epoch_data = epochs.get_data()

    # Get labels
    epoch_labels = epochs.events[:, -1]

    # Get number epochs, number of channels and numbers of points times
    n_epochs, n_channels, n_times = epoch_data.shape

    # Reshape epochs data by 2D Array : n_epochs, n_channels*n_times
    epoch_data = epoch_data.reshape((n_epochs, n_channels * n_times))

    return epoch_data, epoch_labels
