import numpy as np
import pywt


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


# def apply_fourier_transform(raw):
#     """
#     This function apply fourier transform with numpy (np.fft.fft).
#     And it generates a range between low and high frequency + remove frequency outside the range
#
#     Parameters
#     ----------
#     raw : mne.io.Raw
#         An MNE Raw object containing the EEG data with annotations.
#
#     Returns
#     -------
#     filtereed_dataL
#     """
#     data = raw._data
#
#     # filtered_data = raw.copy().filter(l_freq=8., h_freq=40.)
#     # Fourier transform to get frequency need to be remove
#     fft = np.fft.fft(data, axis=1)
#     freq = np.fft.fftfreq(data.shape[1], d=1 / raw.info['sfreq'])
#     idx = np.where((freq <= raw.info['highpass']) & (freq >= raw.info['lowpass']))
#     fft = fft[:, idx[0]]
#
#     # Inverse fourier transform to apply to filtered data
#     filtered_data = np.fft.ifft(fft, axis=1).real
#
#     raw._data = filtered_data
#
#     return raw
