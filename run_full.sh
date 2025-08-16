#!/usr/bin/env bash
set -e
python -m src.train_eval --csv data/reviews.csv --full
