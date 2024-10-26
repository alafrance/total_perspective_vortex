import warnings
from scripts.cli_args import parse_cli_arguments
from src.train import eeg_train


def main():
    defaults = {
        'mode': 'predict',
        'subjects': list(range(1, 110)),
        'experiments': [3, 4, 7, 8, 11, 12],
        'force_train': False,
        'data_dir': "./data/raw/",
        'output_model_file': './data/models/eeg_model.pkl',
        'no_pickle': False
    }
    filter_warnings()
    args = parse_cli_arguments(defaults)
    try:
        if args.mode == "train":
            eeg_train(subjects=args.subjects,
                      experiments=args.experiments,
                      data_dir=args.data_dir,
                      output_model_file=args.output_model_file,
                      force_train=args.force_train,
                      no_pickle=args.no_pickle)
        elif args.mode == "predict":
            print("Predict bip bip bip...")
        else:
            raise ValueError("Error Mode: Train or predict")
    except KeyboardInterrupt:
        print("BCI EEG Interrupt")
    except ValueError as e:
        print(e)
    except FileNotFoundError as e:
        print(f'{e.args[1]}: {e.filename}')


def filter_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)


if __name__ == '__main__':
    main()
