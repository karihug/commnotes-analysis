import os
import glob
import pandas as pd
from typing import Tuple, Dict, Any
from tqdm import tqdm
import re

def get_dtype_mapping() -> Dict[str, Dict[str, Any]]:
    """
    Returns predefined dtype mappings for large TSV files to optimize memory usage.
    """
    return {
        'scoredNotes.tsv': {
            'noteId': 'int', 'tweetId': 'string', 'classification': 'category',
            'noteFactor1': 'float32', 'noteIntercept': 'float32', 'createdAtMillis': 'int64',
        },
        'note_params.tsv': {
            'noteId': 'int', 'tweetId': 'string', 'classification': 'category',
            'noteFactor1': 'float32', 'noteIntercept': 'float32', 'createdAtMillis': 'int64',
        },
        'rater_model_output.tsv': {
            'participantId': 'string', 'raterFactor1': 'float32', 'raterIntercept': 'float32',
        },
        'rater_params.tsv': {
            'participantId': 'string', 'raterFactor1': 'float32', 'raterIntercept': 'float32',
        },
        'note_status_history.tsv': {
            'noteId': 'int', 'noteAuthorParticipantId': 'string', 'tweetId': 'string',
            'createdAtMillis': 'int64', 
        },
        'helpfulness_scores.tsv': { 
            'raterParticipantId': 'string', 'noteId': 'int', 'helpfulnessScore': 'float32',
            'raterFactor1': 'float32', 'raterIntercept': 'float32',
            'crhCrnhRatioDifference': 'float32', 'meanNoteScore': 'float32',
            'raterAgreeRatio': 'float32', 'aboveHelpfulnessThreshold': 'boolean',
            'isEmergingWriter': 'boolean', 'enrollmentState': 'category'
        },
        'auxiliary_note_info.tsv': { 'noteId': 'int' }
    }

def load_weekly_outputs(base_dir: str, debug: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads weekly Community Notes output files from a specified directory.
    This version loads ALL columns to be robust against schema changes.
    """
    pattern = os.path.join(base_dir, "output_*")
    dtype_mappings = get_dtype_mapping()
    
    data_lists = {'note': [], 'rater': [], 'status': [], 'helpfulness': [], 'aux': [], "aux_rater":[]}
    file_keys = {
        'note': ['scoredNotes.tsv', 'note_params.tsv'],
        'rater': ['rater_model_output.tsv', 'rater_params.tsv'],
        'status': ['note_status_history.tsv'],
        'helpfulness': ['helpfulness_scores.tsv'],
        'aux': ['auxiliary_note_info.tsv'],
        # 'aux_rater': ['rater_params.tsv']
    }

    sorted_dirs = sorted(glob.glob(pattern))
    if debug and not sorted_dirs:
        print(f"DEBUG: No directories found matching pattern: {pattern}")

    for week, week_dir in enumerate(tqdm(sorted_dirs, desc="Loading weekly outputs")):
        week_str = os.path.basename(week_dir).replace("output_", "")
        try:
            week_date = pd.to_datetime(week_str, format="%Y_%m_%d")
        except ValueError:
            week_date = pd.to_datetime(week_str)
        
        for key, filenames in file_keys.items():
            file_found_for_key = False
            for filename in filenames:
                file_path = os.path.join(week_dir, filename)
                if debug:
                    print(f"DEBUG: Checking for '{key}' file at: {file_path}")
                
                if os.path.exists(file_path):
                    if debug:
                        print(f"  -> FOUND!")
                    try:
                        # **FIX**: Load all columns, but apply dtypes for known columns.
                        # This avoids errors when columns are missing in some files.
                        df = pd.read_csv(
                            file_path, 
                            sep='\\t', 
                            engine='python',
                            dtype=dtype_mappings.get(filename) # Use .get() for safety
                        )
                        df['week'] = week
                        df['week_dt'] = week_date
                        if key=="status":
                                # only save df that have current status-nonzero.
                                df=df[~df["currentStatus"].isna()] 
                        data_lists[key].append(df)
                        file_found_for_key = True
                    except Exception as e:
                        print(f"Warning: Could not load {file_path}. Error: {e}")
            
            if not file_found_for_key and debug and key == 'status':
                print(f"  -> NOT FOUND in {week_dir}")

    if debug:
        print((data_lists["status"][0].columns))
        print((data_lists["status"][-1].columns))
    final_dfs = []
    for key in file_keys.keys():
        print(key+" is loaded.")
        if data_lists[key]:
            final_dfs.append(pd.concat(data_lists[key], ignore_index=True))
        else:
            print("key without data",key) 
            final_dfs.append(pd.DataFrame())
    
    return tuple(final_dfs)

def read_ratings_from_directory(ratings_dir: str, start_date_str: str, end_date_str: str) -> pd.DataFrame:
    """
    Reads ratings files by processing them in chunks to avoid MemoryError.
    """
    pattern = os.path.join(ratings_dir, "clean-ratings-*")
    all_rating_files = glob.glob(pattern)
    print(f"Found {len(all_rating_files)} total rating files. Processing them in chunks to filter by date.")

    start_ts = pd.to_datetime(start_date_str).value // 1_000_000
    end_ts = pd.to_datetime(end_date_str).value // 1_000_000

    essential_cols = {
        'raterParticipantId': 'string', 'noteId': 'int64',
        'createdAtMillis': 'int64', 'helpfulnessLevel': 'category'
    }

    df_list = []
    for path in tqdm(all_rating_files, desc="Reading ratings"):
        try:
            chunk_iterator = pd.read_csv(
                path, sep="\\t", engine='python', dtype=essential_cols,
                usecols=list(essential_cols.keys()), # kh removed this 1-15-26 for the topic modeling purposes
                chunksize=500000
            )
            for chunk in chunk_iterator:
                filtered_chunk = chunk[(chunk['createdAtMillis'] >= start_ts) & (chunk['createdAtMillis'] <= end_ts)]
                if not filtered_chunk.empty:
                    df_list.append(filtered_chunk)
        except Exception as e:
            print(f"Warning: Could not load or process {path}. Error: {e}")
    
    if not df_list:
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)

def add_helpful_num(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a numeric 'helpfulNum' column based on 'helpfulnessLevel'.
    """
    ratings_df = ratings_df.copy()
    if 'helpfulnessLevel' in ratings_df.columns:
        conditions = {'HELPFUL': 1.0, 'SOMEWHAT_HELPFUL': 0.5, 'NOT_HELPFUL': 0.0}
        ratings_df['helpfulNum'] = ratings_df['helpfulnessLevel'].map(conditions).astype(float)
    return ratings_df


import os
import sys
import pandas as pd
from datetime import datetime

def process_note_topics(df_notes: pd.DataFrame, commnotes_path: str = '/accounts/projects/jchayes/commnotes', output_file: str = 'df_notes_with_topics.csv'):
    """
    Calculates note topics using the TopicModel and merges them into the notes DataFrame.

    This function changes the current working directory to the location of the scoring
    module, imports the necessary class, and then changes back. It uses the TopicModel
    to get topics for the notes and merges the results back into the input DataFrame.

    Args:
        df_notes (pd.DataFrame): The input DataFrame containing note data.
        commnotes_path (str): The base path to the commnotes project. Defaults
                              to '/accounts/projects/jchayes/commnotes'.

    Returns:
        pd.DataFrame: The original DataFrame with an added column for note topics.
    """
    # Change to the directory containing the 'scoring' module to allow the import
    os.chdir('/scratch/users/commnotes/communitynotes/sourcecode')
    try:
        from scoring.topic_model import TopicModel
        # Change back to the working directory
        os.chdir(commnotes_path)
    except ImportError as e:
        print(f"Failed to import TopicModel. Ensure scoring/topic_model.py is in the specified path: {e}")
        # Change back in case of an error
        os.chdir(commnotes_path)
        return None

    # Load the TopicModel
    model = TopicModel()
    
    # Get note topics and merge them into the DataFrame
    note_topics = model.get_note_topics(df_notes)
    df_notes = df_notes.merge(note_topics, on='noteId').copy()
    # Save the processed DataFrame to a CSV file
    df_notes.to_csv(output_file, index=False)
    print(f"DataFrame with topics saved to {output_file}")
    return df_notes