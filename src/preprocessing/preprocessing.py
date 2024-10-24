import os
from src.preprocessing.interface_eeg import load_specific_eeg, extract_epochs, EEGData
from src.preprocessing.signal_transformations import apply_wavelet_transform, apply_fourier_transform
from src.preprocessing.utils import get_subject_name_by_id, get_experiment_name_by_id, get_data_from_pickle_file, \
    save_data_into_pickle_file
from tqdm import tqdm


def pipeline_preprocessed(
        raw_dir='../data/raw/',
        subjects=list(range(1, 110)),
        experiments=[3, 4, 7, 8, 11, 12],
        processed_dir='../data/processed',
        filename_process='preprocessed_data.pkl',
        filename_raw='raw_data.pkl',

):
    data = load_data_and_signal_transform(raw_dir=raw_dir,
                     filename_raw=filename_raw,
                     subjects=subjects,
                     experiments=experiments)
    data_preprocessed = preprocessed_data(data, processed_dir, filename_process)
    return data_preprocessed


def load_data_and_signal_transform(
        raw_dir,
        filename_raw,
        subjects,
        experiments
):
    """
    Load EEG data by subjects and experiments
    Parameters
    ----------
    filename_raw: str
        Filename which the data will be saved
    raw_dir: str
        Directory which can find all raw datas
    experiments: array of int
        Array of experiments
    subjects: range
        Range of subjects
    Returns
    -------
    data: list of EEGData
        All my datas information
    """

    if os.path.exists(os.path.join(raw_dir, filename_raw)):
        return get_data_from_pickle_file(os.path.join(raw_dir, filename_raw),
                                         desc="Subject loaded",
                                         postfix=f'{len(subjects)}/{len(subjects)}')
    data = []
    bar = tqdm(subjects, desc='Loading subject', dynamic_ncols=True, unit='M', unit_scale=True)
    for subject_id in bar:
        subject_name = get_subject_name_by_id(subject_id)
        bar.set_description(f"Loading subject {subject_name}")
        for experiment_id in experiments:
            try:
                experiment_name = get_experiment_name_by_id(experiment_id)

                # EEG info
                raw = load_specific_eeg(raw_dir, subject_name, experiment_name)

                # raw.plot(scalings='auto', title='Raw EEG data')
                # Preprocessing signal transformation
                # raw = apply_fourier_transform(raw)
                # TODO: I need to check bit I think fourier + wavelet transform not a good idea
                #  but fourier just remove high and low freq so not dangerous and less MO pickle but wavelet have a good signal at the end
                raw = apply_wavelet_transform(raw)
                # raw.plot(scalings='auto', title='Raw EEG data filtered')

                # Extract epochs from filtered raw
                epoch_data, epoch_labels = extract_epochs(raw)

                # Append my data
                data.append(EEGData(
                    epoch_data=epoch_data,
                    epoch_labels=epoch_labels,
                    subject_id=subject_id,
                    experiment_id=experiment_id
                ))
            except Exception as e:
                print(f'Error processing subject {subject_name}, experiment {experiment_name}: {e}')
    save_data_into_pickle_file(data, os.path.join(raw_dir, filename_raw))
    return data


def preprocessed_data(raw_data: EEGData, processed_dir, filename_process):
    max_dim = max(eegdata.epoch_data.shape[1] for eegdata in raw_data)

    for epoch_data, epoch_labels, subject_id, experiment_id in raw_data:
        pass

    # max dimension
    #                 n_epochs, n_features = epoch_data.shape
    #                 max_dim = max(max_dim, n_features)
    #                 # Signal transformations


    return raw_data
