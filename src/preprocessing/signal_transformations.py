import numpy as np
import pywt


def apply_fourier_transform(raw):
    """
    This function apply fourier transform with numpy (np.fft.fft).
    And it generates a range between low and high frequency + remove frequency outside the range

    Parameters
    ----------
    raw : mne.io.Raw
        An MNE Raw object containing the EEG data with annotations.

    Returns
    -------
    freq list
        Frequency Range filtered
    magnitude int
        Complex magnitude
    """
    data = raw.get_data()
    fft = np.fft.fft(data, axis=1)
    freq = np.fft.fftfreq(data.shape[1], d=1 / raw.info['sfreq'])
    idx = np.where((freq >= raw.info['highpass']) & (freq <= raw.info['lowpass']))
    freq = freq[idx]
    fft = fft[:, idx[0]]
    return freq, np.abs(fft)


def apply_wavelet_transform(raw, wavelet='db4', level=5):
    """
    This function use the pywt library to apply wavelet transform by using 'ondelette' signal.
    The wavelet transform is more precise than fourier transform because the output frequence are localised with time

    Parameters
    ----------
    raw: array_like
        The data eeg
    wavelet: Wavelet object or name string, optional
        'db4' 'ondelette' family naming 'Daubechies' wuth 4 coefficient's filter. This 'ondelette' is often used in EEG analysis.
    level: int, optional
        level of decomposition (more is high, more is precise)
    Returns
    -------
    coeffs : list
        Ordered list of coefficients arrays
        Useful to rebuild the signal or extract specific characteristic
    """
    data = raw.get_data()
    coeffs = pywt.wavedec(data, wavelet=wavelet, level=level, axis=1)

    return coeffs
