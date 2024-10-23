import joblib
import os


def get_subject_name_by_id(subject_id):
    if subject_id <= 0 or subject_id > 109 or not isinstance(subject_id, int):
        raise ValueError('Subject ID must be between 1 and 109 inclusive')
    if subject_id < 10:
        return f'S00{subject_id}'
    elif subject_id < 100:
        return f'S0{subject_id}'
    else:
        return f'S{subject_id}'
    pass


def get_experiment_name_by_id(experiment_id):
    if experiment_id <= 0 or experiment_id > 14 or not isinstance(experiment_id, int):
        raise ValueError('Experiment ID must be between 1 and 14 inclusive')
    if experiment_id < 10:
        return f'R0{experiment_id}'
    else:
        return f'R{experiment_id}'


def save_data_into_pickle_file(data, path):
    joblib.dump(data, path)


def get_data_from_pickle_file(path):
    return joblib.load(path)


def load_or_launch_func(filepath, func, *args, **kwargs):
    if os.path.exists(filepath):
        data = get_data_from_pickle_file(filepath)
    else:
        # print(args[1])
        # print(**kwargs)
        data = func(args[0], args[1])
        # data = []
        data = []
        # save_data_into_pickle_file(data, filepath)
    return data
