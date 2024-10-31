import pickle
import joblib
import os
from tqdm import tqdm


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
