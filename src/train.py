from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.loader import load_data
# from src.CustomCSP import CustomCSP


def eeg_bci_train(data_dir,
                  subjects,
                  experiments):
    epochs, labels = load_data(data_dir=data_dir, subjects=subjects, experiments=experiments)
    epochs_data_train = epochs.get_data(copy=False)

    param_grid = {
        'CSP__n_components': [2, 4, 6],
        'LDA__solver': ['svd']
    }

    clf = Pipeline([
        # ('CSP', CustomCSP()),
        ('CSP', CSP()),
        ('LDA', LinearDiscriminantAnalysis())
    ])

    print("\n" + "="*50)
    print("Grid search cross validation starting...")
    print("="*50 + "\n")

    grid = GridSearchCV(clf, param_grid, cv=5, verbose=2)
    grid.fit(epochs_data_train, labels)
    best_model = grid.best_estimator_

    print("\n" + "="*50)
    print("Grid search cross validation finished !")
    print(f"Best parameter : {grid.best_params_}")
    print(f"Score : {grid.best_score_}")
    print("="*50 + "\n")

    return best_model

