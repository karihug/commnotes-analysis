import pandas as pd
import argparse
from typing import Optional
import torch
from scoring.pandas_utils import patch_pandas
from scoring.process_data import preprocess_data
from scoring.mf_base_scorer import run_single_round_mf, get_ratings_for_stable_init
import scoring.constants as c
import glob
import os
from datetime import datetime, timedelta

def read_ratings_from_directory(ratings_dir):
    ratings_files = glob.glob(os.path.join(ratings_dir, '*.tsv'))
    df_list = [
        pd.read_csv(f, sep='\t', dtype=c.ratingTSVTypeMapping)
        for f in ratings_files
    ]
    return pd.concat(df_list, ignore_index=True)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

@patch_pandas
def main(args):
    notes = pd.read_csv("notes-00000.tsv", sep='\t', dtype=c.noteTSVTypeMapping)
    noteStatusHistory = pd.read_csv("noteStatusHistory-00000.tsv", sep='\t', dtype=c.noteStatusHistoryTSVTypeMapping)
    userEnrollment = pd.read_csv("userEnrollment-00000.tsv", sep='\t', dtype=c.userEnrollmentTSVTypeMapping)
    ratings = read_ratings_from_directory("ratings")

    # Explicitly enforce data types for merging
    ratings['noteId'] = ratings['noteId'].astype('int64')
    ratings['createdAtMillis'] = ratings['createdAtMillis'].astype('int64')
    notes['createdAtMillis'] = notes['createdAtMillis'].astype('int64')
    noteStatusHistory['createdAtMillis'] = noteStatusHistory['createdAtMillis'].astype('int64')

    start_date = datetime(2022, 8, 1)
    end_date = datetime(2023, 6, 1)
    current_date = start_date

    while current_date <= end_date:
        cutoffTimestampMillis = int(current_date.timestamp() * 1000)

        print(f"\nProcessing week ending on: {current_date.strftime('%Y-%m-%d')}")

        _, ratingsProcessed, _ = preprocess_data(notes, ratings, noteStatusHistory)

        noteParams, raterParams, globalIntercept = run_single_round_mf(
            ratingsProcessed,
            userEnrollment,
            cutoffTimestampMillis=cutoffTimestampMillis,
            seed=42
        )

        # Create and save outputs in a nicely named weekly folder
        folder_name = f"output_{current_date.strftime('%Y_%m_%d')}"
        ensure_dir(folder_name)

        noteParams.to_csv(os.path.join(folder_name, "note_params.tsv"), sep='\t', index=False)
        raterParams.to_csv(os.path.join(folder_name, "rater_params.tsv"), sep='\t', index=False)

        print(f"Saved results to {folder_name}")
        print(f"Global Intercept for {current_date.strftime('%Y-%m-%d')}: {globalIntercept}")

        current_date += timedelta(days=7)  # Increment by one week

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enforce-types", action="store_true")
    args = parser.parse_args([])
    args.enforce_types = False
    main(args)

