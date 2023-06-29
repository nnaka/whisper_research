#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    $ spark-submit --deploy-mode client _.py
"""
from argparse import ArgumentParser
from enum import Enum
import os
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset
from transformers import pipeline, Pipeline


class EntailmentCategory(Enum):
    CONTRADICTION = "CONTRADICTION"
    ENTAILMENT = "ENTAILMENT"
    NEUTRAL = "NEUTRAL"


# def main(spark: SparkSession, is_full: bool, is_final: bool) -> None:
def main(is_full: bool, is_final: bool) -> None:
    """Main routine
    Parameters
    ----------
    spark : SparkSession object
    """
    print("Running research project")

    # Model source: https://huggingface.co/roberta-large-mnli
    classifier: Pipeline = pipeline("text-classification", model="roberta-large-mnli")

    # Test zero-shot NLI classification
    print(
        classifier(
            "A soccer game with multiple males playing. Some men are playing a sport."
        )
    )
    ## [{'label': 'ENTAILMENT', 'score': 0.98}]

    # Dataset source: https://huggingface.co/datasets/openwebtext
    dataset: Dataset = load_dataset("openwebtext", split="train")

    # Get entailment examples
    results: Dict[EntailmentCategory, List[str]] = {
        "CONTRADICTION": [],
        "ENTAILMENT": [],
        "NEUTRAL": [],
    }
    label: str = ""
    score: float = 0.0
    for data in dataset[:100]:
        label, score = classifier(data)[0]
        if score > 0.5:
            results[label].append(data)

    # import pdb; pdb.set_trace()
    abr_results = {k: len(v) for k, v in results.items()}
    print(abr_results)


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument(
        "--full", dest="is_full", action="store_true", help="Run on full dataset"
    )
    parser.add_argument(
        "--final",
        dest="is_final",
        action="store_true",
        help="Run on final datasets, train/test",
    )

    args = parser.parse_args()
    print(f"Using args: {args}")

    # Create the spark session object
    # spark = SparkSession.builder.appName("final_project").getOrCreate()

    # Call our main routine
    # main(spark, args.is_full, args.is_final)
    main(args.is_full, args.is_final)
