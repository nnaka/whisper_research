#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import csv
from enum import Enum
import functools
import os
import sys
from typing import Dict, Generator, List, Optional, Tuple, Union

# Must be called before the import of transformers etc to properly set the .cache dir
def setup_env(path: str) -> None:
    """Modifying where the .cache directory is getting stored"""
    os.environ["HF_HOME"] = path
    os.environ["TORCH_HOME"] = path
    os.environ["TRANSFORMERS_CACHE"] = path
    print(
        f"Environment variables set TORCH_HOME = {os.environ['TORCH_HOME']}; HF_HOME={os.environ['HF_HOME']}; TRANSFORMERS_CACHE={os.environ['TRANSFORMERS_CACHE']}"
    )


setup_env("/scratch/nn1331/entailment/.cache")

from datasets import Dataset, load_dataset
import evaluate
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    pipeline,
    Pipeline,
    TrainingArguments,
    Trainer,
)


def main(is_full: bool, is_final: bool) -> None:
    """Main routine"""
    run_zero_shot()


def run_zero_shot() -> None:
    print("Running research project")

    # Model source: https://huggingface.co/roberta-large-mnli
    classifier: Pipeline = pipeline("text-classification", model="roberta-large-mnli")

    # Test zero-shot NLI classification
    print(
        classifier(
            "A soccer game with multiple males playing. Some men are playing a sport."
        )
    )
    # [{'label': 'ENTAILMENT', 'score': 0.98}]

    open_web_text(classifier)


def get_last_sentences(text: str, max_chars: int) -> str:
    sentences: List[str] = text.split(". ")  # Split the text into sentences
    selected_sentences: List[str] = []
    total_chars: int = 0

    # Iterate over the sentences in reverse order
    for sentence in reversed(sentences):
        selected_sentences.insert(0, sentence)  # Insert sentence at the beginning

        # Calculate the total characters
        total_chars += len(sentence) + 2  # Add 2 for the period and space

        if total_chars >= max_chars:
            break

    return ". ".join(selected_sentences)


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
    parser.add_argument(
        "--out",
        dest="output",
        type=str,
        help="Output CSV file path",
    )

    args = parser.parse_args()
    print(f"Using args: {args}")

    # Create the spark session object
    # spark = SparkSession.builder.appName("final_project").getOrCreate()

    # Call our main routine
    # main(spark, args.is_full, args.is_final)
    main(args.is_full, args.is_final, args.output)
