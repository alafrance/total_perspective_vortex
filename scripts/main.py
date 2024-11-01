import os.path
import warnings
from scripts.cli_args import parse_cli_arguments
from src.utils_pickle import save_data_into_pickle_file, get_data_from_pickle_file
from src import eeg_bci_train
from src import eeg_bci_predict


def main():
    defaults = {
        'mode': 'train',
        'subjects': list(range(1, 30)),
        # 'experiments': [3, 4, 7, 8, 11, 12],  # Task 1, 2
        'experiments': [5, 6, 9, 10, 13, 14],  # Task 3, 4
        'force_train': False,
        'data_dir': "../data/raw/",
        'process_dir': "../data/processed/",
        'model_file': '../data/models/eeg_model.pkl',
        'no_pickle': False,
        'bonus_dataset': False
    }
    filter_warnings()
    args = parse_cli_arguments(defaults)
    try:
        if args.mode == "train":
            if os.path.exists(args.model_file) and args.force_train:
                os.remove(args.model_file)
            model = eeg_bci_train(subjects=args.subjects,
                                  experiments=args.experiments,
                                  data_dir=args.data_dir,
                                  process_dir=args.process_dir,
                                  bonus_dataset=args.bonus_dataset)
            if not args.no_pickle:
                save_data_into_pickle_file(model, args.model_file, postfix="Model", desc="Model saved")
        if args.mode == "predict" or args.mode == 'all':
            model = get_data_from_pickle_file(args.model_file, postfix="Model", desc="Model loaded")
            eeg_bci_predict(subjects=args.subjects,
                            experiments=args.experiments,
                            data_dir=args.data_dir,
                            model=model,
                            process_dir=args.process_dir,
                            bonus_dataset=args.bonus_dataset)
    except KeyboardInterrupt:
        print("BCI EEG Interrupt")
    except FileNotFoundError as e:
        print(f'{e.args[1]}: {e.filename}')


def filter_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)


if __name__ == '__main__':
    main()
