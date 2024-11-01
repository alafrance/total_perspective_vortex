from scipy.io import loadmat
import numpy as np
import mne
from tqdm import tqdm
mne.set_log_level('WARNING')


def load_data_from_bonus_directory(raw_dir):
    range_numbers = ['d']
    epochs = []
    labels = []
    for range_number in tqdm(range_numbers, desc="Loading bonus data"):
        filename = raw_dir+'BCICIV_calib_ds1'+range_number+'.mat'
        epochs_, labels_ = loader_matlab_from_filename(filename)
        labels_ = np.asarray(labels_).astype(int).ravel()
        epochs.append(epochs_)
        labels.append(labels_)
    return mne.concatenate_epochs(epochs), np.concatenate(labels)


def loader_matlab_from_filename(filename):
    matdata = loadmat(filename)

    data, info, events = convert_matdata_to_data_info_events(matdata)
    tmin = 0.5
    event_id = {
        'left': 0,
        'right': 1
    }
    epochs = mne.EpochsArray(data, info, events, tmin, event_id, verbose=None)
    epochs.filter(l_freq=8, h_freq=32)
    labels = epochs.events[:,-1]
    labels = np.array([0 if label == 1 else 1 for label in labels])
    return epochs, labels


def convert_matdata_to_data_info_events(matdata):
    sfreq = matdata['nfo']['fs'][0][0][0][0]
    EEGData = matdata['cnt'].T
    n_channels, n_samples = EEGData.shape
    channels_names = [s[0] for s in matdata['nfo']['clab'][0][0][0]]
    info = mne.create_info(
        ch_names = channels_names,
        ch_types = ['eeg']*n_channels,
        sfreq    = sfreq )
    trials = {}
    win = np.arange(int(0.5*sfreq), int(2.5*sfreq))
    cl_lab = [s[0] for s in matdata['nfo']['classes'][0][0][0]]
    events_codes = matdata['mrk'][0][0][1]
    events_onset = matdata['mrk'][0][0][0]
    cl1    = cl_lab[0]
    cl2    = cl_lab[1]
    nsamples = len(win)

    for cl, code in zip(cl_lab, np.unique(events_codes)):
        cl_onsets = events_onset[events_codes == code]
        trials[cl] = np.zeros((n_channels, nsamples, len(cl_onsets)))
        for i, onset in enumerate(cl_onsets):
            trials[cl][:,:,i] = EEGData[:, win+onset]
    left_hand  = np.rollaxis(trials[cl1], 2, 0)
    right_hand = np.rollaxis(trials[cl2], 2, 0)
    data = np.concatenate([left_hand, right_hand])
    Y = np.concatenate([-np.ones(left_hand.shape[0]),
                        np.ones(right_hand.shape[0])])
    len_events = Y.shape[0]
    ev = [i*sfreq*3 for i in range(len_events)]
    events = np.column_stack((np.array(ev,  dtype = int),
                              np.zeros(len_events,  dtype = int),
                              np.array(Y,  dtype = int)))
    events[events[:, -1] == -1, -1] = 0
    return data, info, events

