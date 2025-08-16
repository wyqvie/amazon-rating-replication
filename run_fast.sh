#!/usr/bin/env bash
set -e
python -m src.train_eval --csv data/reviews.csv --fast --subsample 2000
