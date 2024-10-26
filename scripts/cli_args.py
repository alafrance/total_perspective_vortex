import argparse


def parse_cli_arguments(defaults):
    parser = argparse.ArgumentParser(
        description='Brain Computer Interface for EEG (electroencephalogram) from this data : '
                    'https://physionet.org/content/eegmmidb/1.0.0/)',
        formatter_class=CustomFormatter
    )

    # Mode
    parser.add_argument('mode',
                        choices=['train', 'predict'],
                        default=defaults['mode'],
                        help='Mode of operation: train or predict')

    # Subject or experiments id
    parser.add_argument('--subjects',
                        type=int,
                        nargs='+',
                        default=defaults['subjects'],
                        metavar='ID',
                        help='ID(s) of the subject')
    parser.add_argument('--experiments',
                        type=int,
                        nargs='+',
                        default=defaults['experiments'],
                        metavar='ID',
                        help='ID(s) of the experiment')

    # Train
    parser.add_argument('--force_train',
                        action='store_true',
                        default=defaults['force_train'],
                        help='Force retraining even if pickle exists')

    # Files config
    parser.add_argument('--data_dir',
                        type=str,
                        default=defaults['data_dir'],
                        help="The data directory for training or predict")
    parser.add_argument("--output_model_file",
                        type=str,
                        default=defaults['output_model_file'],
                        help="The output model pickle file")
    parser.add_argument('--no_pickle',
                        action='store_true',
                        default=defaults['no_pickle'],
                        help='Do not save model to pickle file')

    args = parser.parse_args()
    return args


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action):
        help_text = action.help
        if action.default not in (None, argparse.SUPPRESS):
            if isinstance(action.default, list) and len(action.default) > 10:
                default_str = f"[1..{action.default[-1]}]"
                help_text += f' (default: {default_str})'
            else:
                help_text += f' (default: {action.default})'
        return help_text
