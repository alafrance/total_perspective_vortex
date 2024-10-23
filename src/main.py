import argparse
import warnings
from src.preprocessing import pipeline_preprocessed


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # parser = argparse.ArgumentParser(description='BIC for EEG Data')
    pipeline_preprocessed()


if __name__ == '__main__':
    main()
