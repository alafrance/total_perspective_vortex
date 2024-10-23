import os.path

from src.preprocessing.load_data import load_data
from src.preprocessing.utils import load_or_launch_func


def pipeline_preprocessed(
        raw_dir='../data/raw/',
        subjects=list(range(1, 110)),
        experiments=[3, 4, 7, 8, 11, 12],
        processed_dir='../data/processed',
        filename_process='preprocessed_data.pkl',

):

    # raw_path = os.path.join(raw_dir, filename_raw)
    data = load_data(raw_dir, subjects, experiments)
    # if os.path.exists(raw_path):
    #     data = load_data(raw_dir, subjects, experiments)
    #     save_data_into_pickle_file(data, raw_path)
    # else:
    #     data = get_data_from_pickle_file(raw_path)

    # raw_path = os.path.join(raw_dir, filename_raw)
    # if os.path.exists(raw_path):
    #     data_preprocessed = preprocessed_data(data)
    #     save_data_into_pickle_file(data_preprocessed, raw_path)
    # else:
    #     data = get_data_from_pickle_file(raw_path)

    return data


def preprocessed_data(raw_data):
    # max dimension
    #                 n_epochs, n_features = epoch_data.shape
    #                 max_dim = max(max_dim, n_features)
    #                 # Signal transformations
    #                 apply_fourier_transform(filtered_raw)
    #                 apply_wavelet_transform(filtered_raw)
    return []
