# Put the code for your API here.
"""
ML Pipeline
"""
import argparse

import functions.data as data
import functions.model as model
import functions.train_test_model as train_test


import logging


def go(args):
    """
    Execute the pipeline
    """
    logging.basicConfig(level=logging.INFO)

    if args.action == "all" or args.action == "basic_cleaning":
        logging.info("Basic cleaning procedure started")
        data.clean_data()

    if args.action == "all" or args.action == "train_test_model":
        logging.info("Train/Test model procedure started")
        train_test.train_test()

    if args.action == "all" or args.action == "check_score":
        logging.info("Score check procedure started")
        model.score_slices()


if __name__ == "__main__":
    """
    Main entrypoint
    """
    parser = argparse.ArgumentParser(description="ML training pipeline")

    parser.add_argument(
        "--action",
        type=str,
        choices=["basic_cleaning",
                 "train_test_model",
                 "check_score",
                 "all"],
        default="all",
        help="Pipeline action"
    )

    args = parser.parse_args()

    go(args)
