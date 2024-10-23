from src.preprocessing.interface_eeg import load_specific_eeg, extract_epochs
from src.preprocessing.signal_transformations import apply_fourier_transform, apply_wavelet_transform
from src.preprocessing.utils import get_subject_name_by_id, get_experiment_name_by_id
from tqdm import tqdm


def load_data(
    raw_dir,
    subjects,
    experiments
):
    """
    Load EEG data by subjects and experiments
    Parameters
    ----------
    raw_dir: str, optional
        Directory which can find all raw datas
    experiments: array of int, optional
        Array of experiments
    subjects: range, optional,
        Range of subjects
    Returns
    -------
    data: list of EEGData
        All my datas information
    """
    data = []
    bar = tqdm(subjects, desc='Loading subject', dynamic_ncols=True)
    for subject_id in bar:
        subject_name = get_subject_name_by_id(subject_id)
        bar.set_description(f"Loading subject {subject_name}")
        bar.set_postfix_str(f'{subject_id}/{len(subjects)}')
        for experiment_id in experiments:
            try:
                experiment_name = get_experiment_name_by_id(experiment_id)

                # EEG info
                raw, filtered_raw = load_specific_eeg(raw_dir, subject_name, experiment_name)
                epoch_data, epoch_labels = extract_epochs(filtered_raw)

                # Append my data
                data.append((
                    epoch_data,
                    epoch_labels,
                    subject_id,
                    experiment_id
                ))
            except Exception as e:
                print(f'Error processing subject {subject_name}, experiment {experiment_name}: {e}')
    return data


class EEGData:
    def __init__(self, epoch_data, epoch_labels, subject_id, experiment_id):
        """
        Class to contain EEG data structured

        Parameters
        ----------
        epoch_data : numpy.ndarray
            Extract epoch data from EEG signal
        epoch_labels : numpy.ndarray
            Label associated to epoch data
        subject_id : int
            Subject id
        experiment_id : int
            Experiment Id
        """
        self.epoch_data = epoch_data
        self.epoch_labels = epoch_labels
        self.subject_id = subject_id
        self.experiment_id = experiment_id

    def __repr__(self):
        return f"EEGData(subject_id={self.subject_id}, experiment_id={self.experiment_id}, n_epochs={len(self.epoch_data)})"
