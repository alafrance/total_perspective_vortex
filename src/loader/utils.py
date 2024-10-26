import pickle
import joblib
import os
from tqdm import tqdm


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


def save_data_into_pickle_file(data, filename, chunksize=10**6, postfix=None, desc="Saving data"):
    data_bytes = pickle.dumps(data)
    total_size = len(data_bytes)

    with tqdm(total=total_size, desc=desc, unit='M', unit_scale=True) as pbar:
        pbar.set_postfix_str(postfix)
        with open(filename, 'wb') as f:
            for i in range(0, total_size, chunksize):
                chunk = data_bytes[i:i+chunksize]
                f.write(chunk)
                pbar.update(len(chunk))
    joblib.dump(data, filename)


def get_data_from_pickle_file(path, chunksize=10**6, postfix=None, desc="Loading data"):
    total_size = os.path.getsize(path)

    with tqdm(total=total_size, desc=desc, unit='M', unit_scale=True) as pbar:
        pbar.set_postfix_str(postfix)
        data_bytes = bytearray()

        with open(path, 'rb') as f:
            while True:
                chunk = f.read(chunksize)
                if not chunk:
                    break
                data_bytes.extend(chunk)
                pbar.update(len(chunk))

    return joblib.load(path)
