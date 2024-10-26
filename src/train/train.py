import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from src.loader.utils import save_data_into_pickle_file, get_data_from_pickle_file
from src.loader import pipeline_loader_data
# from src.preprocessing.CustomLDA import CustomLDA
from src.train.CustomSCP import CustomSCP
from src.train.CustomStandardScaler import CustomStandardScaler


def eeg_train(subjects,
              experiments,
              data_dir,
              output_model_file,
              force_train,
              no_pickle):
    if os.path.exists(output_model_file) and not force_train:
        return get_data_from_pickle_file(output_model_file, postfix="Model", desc="Model loaded")

    train_val_test_data = pipeline_loader_data(subjects, experiments, data_dir)
    model = cross_validation(train_val_test_data)

    if not no_pickle:
        save_data_into_pickle_file(model, output_model_file, postfix="Model", desc="Model saved")

    return model


def cross_validation(train_val_test_data):
    print("\n" + "="*50)
    print("Cross-Validation Process")
    print(f"Data shape: {train_val_test_data['X_train'].shape}")
    print(f"Number of classes: {len(np.unique(train_val_test_data['y_train']))}")
    print("="*50 + "\n")
    print(f'Initializing validation settings...')

    pipeline, hyperparameter_grid = config_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(balanced_accuracy_score)

    print(f'\nStarting grid search cross validation')
    grid_search = GridSearchCV(
        pipeline,
        hyperparameter_grid,
        cv=cv,
        scoring=scorer,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(train_val_test_data['X_train'], train_val_test_data['y_train'])

    print("\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")

    X_combined = np.concatenate((train_val_test_data['X_train'], train_val_test_data['X_val']))
    y_combined = np.concatenate((train_val_test_data['y_train'], train_val_test_data['y_val']))

    val_score = grid_search.best_estimator_.score(train_val_test_data['X_val'], train_val_test_data['y_val'])
    print(f"\nValidation score: {val_score:.4f}")

    print("Final evaluation...")
    best_model = grid_search.best_estimator_
    best_model.fit(X_combined, y_combined)

    print("\n" + "="*50)
    print("Cross-Validation Process")
    print(f"Finish !")
    print("="*50 + "\n")

    return best_model


def config_pipeline():
    pipeline = Pipeline([
        ('scaler', CustomStandardScaler()),
        ('scp', CustomSCP()),
        ('classifier', LinearDiscriminantAnalysis())
    ])
    hyperparameter_grid = [
        {
            'scp__n_components': [6, 8, 10, 12],
            'classifier__solver': ['svd'],
            'classifier__shrinkage': [None]
        }
    ]
    return pipeline, hyperparameter_grid
