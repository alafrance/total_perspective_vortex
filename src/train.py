from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.loader import load_data
from src.CustomCSP import CustomCSP
from src.loader_bonus import load_data_from_bonus_directory
import os

from src.utils_pickle import save_data_into_pickle_file


def eeg_bci_train(data_dir,
                  subjects,
                  experiments,
                  process_dir,
                  bonus_dataset):
    if bonus_dataset:
        epochs, labels = load_data_from_bonus_directory(data_dir)
        X_train, X_test, y_train, y_test = train_test_split(epochs.get_data(copy=False), labels, test_size=0.2, random_state=42)
        processed_path = os.path.join(process_dir, 'test_data_bonus_dataset.pkl')
        save_data_into_pickle_file((X_test, y_test), processed_path, desc="Test data saved")
    else:
        epochs, labels = load_data(data_dir=data_dir, subjects=subjects, experiments=experiments)
        X_train = epochs.get_data(copy=False)
        y_train = labels

    param_grid = {
        'CSP__n_components': [2, 4, 6],
        'LDA__solver': ['svd']
    }

    clf = Pipeline([
        ('CSP', CustomCSP()),
        ('LDA', LinearDiscriminantAnalysis())
    ])

    print("\n" + "="*50)
    print("Grid search cross validation starting...")
    print("="*50 + "\n")

    grid = GridSearchCV(clf, param_grid, cv=5, verbose=2)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print("\n" + "="*50)
    print("Grid search cross validation finished !")
    print(f"Best parameter : {grid.best_params_}")
    print(f"Score : {grid.best_score_}")
    print("="*50 + "\n")

    return best_model

