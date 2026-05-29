#!/usr/bin/env python
"""
Batch script for running matrix factorization analysis.
Converts run_mf.ipynb to a standalone Python script.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for batch processing
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
from datetime import datetime

from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
import statsmodels.formula.api as smf

# Change to analysis directory
os.chdir('/accounts/projects/jchayes/commnotes/analysis/')

from src.data_loader import *
from src.analysis import *
from src.visualization import *
from src.utils import *

import matrix_factorization
import constants as c
import utility as utility

# --- Configuration ---
NOTES_PATH = "/scratch/users/commnotes/communitynotes/sourcecode/notes-00000.tsv"
HISTORY_PATH = "/scratch/users/commnotes/communitynotes/sourcecode/noteStatusHistory-00000.tsv"
RATINGS_DIR = "/scratch/users/commnotes/communitynotes/sourcecode/ratings"
PAPER_DATA_URL = "https://raw.githubusercontent.com/trenault/CommunityNotes/main/database_replication.csv"
WEEKLY_OUTPUTS_PATH = "/scratch/users/commnotes/communitynotes2022/communitynotes/static/sourcecode/weekly_outputs"
WEEKLY_OUTPUTS_PATH_2025 = "/scratch/users/commnotes/communitynotes/sourcecode/weekly_outputs"

max_date = "2024-06-01"
start_date = "2023-06-01"

# Timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def compute_user_variance(ratings_df):
    """Compute variance of residuals for each user."""
    
    ratings_df = ratings_df.copy()
    ratings_df['residual'] = ratings_df['helpfulNum'] - ratings_df['estimatedHelpfulness']
    user_variances = ratings_df.groupby('raterParticipantId')['residual'].var().reset_index()
    user_variances.columns = ['raterParticipantId', 'helpfulnessVariance']
    return user_variances


def run_weekly_analysis(output_dir):
    """Run one-week ahead estimates."""
    print("\n" + "="*80)
    print("RUNNING ONE-WEEK AHEAD ESTIMATES")
    print("="*80 + "\n")
    
    # Load ratings data
    print(f"Loading ratings data from {start_date} to {max_date}...")
    ratings_df = read_ratings_from_directory(RATINGS_DIR, start_date_str="2023-01-01", end_date_str=max_date)
    
    # Convert helpfulnessLevel to numeric helpfulNum
    ratings_df['helpfulNum'] = None
    ratings_df.loc[ratings_df['helpfulnessLevel'] == 'HELPFUL', 'helpfulNum'] = 1.0
    ratings_df.loc[ratings_df['helpfulnessLevel'] == 'SOMEWHAT_HELPFUL', 'helpfulNum'] = 0.5
    ratings_df.loc[ratings_df['helpfulnessLevel'] == 'NOT_HELPFUL', 'helpfulNum'] = 0.0
    
    # Drop rows where helpfulNum is NaN
    ratings_df = ratings_df[ratings_df['helpfulNum'].notna()]
    ratings_df['helpfulNum'] = ratings_df['helpfulNum'].astype(float)
    
    # Modify date variables
    ratings_df['createdAtDate'] = pd.to_datetime(ratings_df['createdAtMillis'], unit='ms').dt.normalize()
    ratings_df['createdAtWeek'] = ratings_df['createdAtDate'] - pd.to_timedelta(ratings_df['createdAtDate'].dt.dayofweek, unit='d')
    
    print(f"Total ratings loaded: {len(ratings_df)}")

    
    # Initialize storage
    noteParamsByWeek_stage1 = pd.DataFrame()
    raterParamsByWeek_stage1 = pd.DataFrame()
    noteParamsByWeek_stage2 = pd.DataFrame()
    raterParamsByWeek_stage2 = pd.DataFrame()

    globalBiasByWeek_stage1 = []
    globalBiasByWeek_stage2 = []
    
    # Get all weeks after the start date
    weeks = sorted(ratings_df[ratings_df['createdAtWeek'] > start_date]['createdAtWeek'].unique())

    # First, train on historical data (up to start_date) to get initial factors
    print(f"\nTraining on historical data up to {start_date}...")
    historical_data = ratings_df[ratings_df['createdAtWeek'] <= start_date]
    historical_data = utility.filter_ratings(historical_data)
    
    # Initialize with historical data
    noteParams_prev, raterParams_prev, globalBias_prev = matrix_factorization.run_mf(
        historical_data,
        c.l2_lambda,
        c.l2_intercept_multiplier,
        c.numFactors,
        c.epochs,
        c.useGlobalIntercept,
        runName=f"historical_up_to_{start_date}"
    )
    
    # Compute initial user variances on historical data
    historical_data['raterFactor1'] = historical_data['raterParticipantId'].map(raterParams_prev.set_index('raterParticipantId')['raterFactor1'])
    historical_data['raterIntercept'] = historical_data['raterParticipantId'].map(raterParams_prev.set_index('raterParticipantId')['raterIntercept'])
    historical_data['noteFactor1'] = historical_data['noteId'].map(noteParams_prev.set_index('noteId')['noteFactor1'])
    historical_data['noteIntercept'] = historical_data['noteId'].map(noteParams_prev.set_index('noteId')['noteIntercept'])
    historical_data['estimatedHelpfulness'] = (historical_data['raterFactor1'] * historical_data['noteFactor1'] + 
                                               historical_data['raterIntercept'] + historical_data['noteIntercept'] + 
                                               globalBias_prev[0][0].detach().numpy())
    user_vars_prev = compute_user_variance(historical_data)
    user_vars_prev['helpfulnessVariance'] = user_vars_prev['helpfulnessVariance'].clip(lower=1e-4)
    
    # Iterate through each week
    print(f"\nProcessing {len(weeks)} weeks...")
    for i, week in enumerate(weeks):
        print(f"\n[{i+1}/{len(weeks)}] Processing week: {week.date()}")
        
        # Get cumulative data up to current week and filter it
        cumul_week_data = ratings_df[ratings_df['createdAtWeek'] <= week]
        cumul_week_data = utility.filter_ratings(cumul_week_data)
        
        # Now get only the current week's data from the filtered cumulative data
        # week_data = cumul_week_data[cumul_week_data['createdAtWeek'] == week]
        
        # STAGE 1: Run MF on current week's data (after filtering), warm starting with previous week's factors
        print(f"  Stage 1: Running MF on {len(cumul_week_data)} ratings...")
        noteParams, raterParams, globalBias = matrix_factorization.run_mf(
            cumul_week_data,
            c.l2_lambda,
            c.l2_intercept_multiplier,
            c.numFactors,
            c.epochs,
            c.useGlobalIntercept,
            noteInit=noteParams_prev,
            userInit=raterParams_prev,
            runName=f"week_{week.date()}_stage1"
        )

        globalBiasByWeek_stage1.append({
            'createdAtWeek': week,
            'globalBias': globalBias[0][0].detach().numpy()
        })
        
        # Combine current week's factors with previous factors for users who didn't appear this week
        # Start with previous week's factors (historical)
        combined_raterParams = raterParams_prev.copy()
        combined_noteParams = noteParams_prev.copy()
        
        # Update with current week's Stage 1 factors (overwrite for users/notes that appeared this week)
        week_users = set(cumul_week_data['raterParticipantId'].unique())
        week_notes = set(cumul_week_data['noteId'].unique())
        
        # Remove users/notes that appeared this week from the combined params
        combined_raterParams = combined_raterParams[~combined_raterParams['raterParticipantId'].isin(week_users)]
        combined_noteParams = combined_noteParams[~combined_noteParams['noteId'].isin(week_notes)]
        
        # Add current week's factors
        combined_raterParams = pd.concat([combined_raterParams, raterParams], ignore_index=True)
        combined_noteParams = pd.concat([combined_noteParams, noteParams], ignore_index=True)
        
        # Map combined factors to cumulative data for variance computation
        cumul_week_data['raterFactor1'] = cumul_week_data['raterParticipantId'].map(combined_raterParams.set_index('raterParticipantId')['raterFactor1'])
        cumul_week_data['raterIntercept'] = cumul_week_data['raterParticipantId'].map(combined_raterParams.set_index('raterParticipantId')['raterIntercept'])
        cumul_week_data['noteFactor1'] = cumul_week_data['noteId'].map(combined_noteParams.set_index('noteId')['noteFactor1'])
        cumul_week_data['noteIntercept'] = cumul_week_data['noteId'].map(combined_noteParams.set_index('noteId')['noteIntercept'])
        cumul_week_data['estimatedHelpfulness'] = (cumul_week_data['raterFactor1'] * cumul_week_data['noteFactor1'] + 
                                                    cumul_week_data['raterIntercept'] + cumul_week_data['noteIntercept'] + 
                                                    globalBias[0][0].detach().numpy())
        
        # Compute user variances on cumulative data for users who appeared this week
        cumul_user_vars = compute_user_variance(cumul_week_data)
        cumul_user_vars['helpfulnessVariance'] = cumul_user_vars['helpfulnessVariance'].fillna(1e-4).clip(lower=1e-4)
        week_user_vars = cumul_user_vars[cumul_user_vars['raterParticipantId'].isin(week_users)]
        
        # Update the overall user variance dictionary: update for users in this week, keep old for others
        user_vars = user_vars_prev.copy()
        # Remove users who appeared this week from the old variances
        user_vars = user_vars[~user_vars['raterParticipantId'].isin(week_users)]
        # Add the newly computed variances for this week's users (computed on cumulative data)
        user_vars = pd.concat([user_vars, week_user_vars], ignore_index=True)
        
        print(f"  User variance stats - Min: {user_vars['helpfulnessVariance'].min():.6f}, "
              f"Mean: {user_vars['helpfulnessVariance'].mean():.6f}, "
              f"Max: {user_vars['helpfulnessVariance'].max():.6f}")
        
        # STAGE 2: Run weighted MF
        print(f"  Stage 2: Running weighted MF...")
        noteParams2, raterParams2, globalBias2 = matrix_factorization.run_mf(
            cumul_week_data,
            c.l2_lambda,
            c.l2_intercept_multiplier,
            c.numFactors,
            c.epochs,
            c.useGlobalIntercept,
            user_variances=user_vars,
            noteInit=noteParams,
            userInit=raterParams,
            runName=f"week_{week.date()}_stage2"
        )

        globalBiasByWeek_stage2.append({
            'createdAtWeek': week,
            'globalBias': globalBias2[0][0].detach().numpy()
        })
        
        # Add week identifier
        noteParams['createdAtWeek'] = week
        raterParams['createdAtWeek'] = week
        noteParams2['createdAtWeek'] = week
        raterParams2['createdAtWeek'] = week

        
        # Append to weekly results
        noteParamsByWeek_stage1 = pd.concat([noteParamsByWeek_stage1, noteParams], ignore_index=True)
        raterParamsByWeek_stage1 = pd.concat([raterParamsByWeek_stage1, raterParams], ignore_index=True)
        noteParamsByWeek_stage2 = pd.concat([noteParamsByWeek_stage2, noteParams2], ignore_index=True)
        raterParamsByWeek_stage2 = pd.concat([raterParamsByWeek_stage2, raterParams2], ignore_index=True)
        
        # Update previous week's factors and variances for next iteration
        noteParams_prev = noteParams
        raterParams_prev = raterParams
        user_vars_prev = user_vars

    # Convert global bias lists to DataFrames
    globalBiasByWeek_stage1 = pd.DataFrame(globalBiasByWeek_stage1)
    globalBiasByWeek_stage2 = pd.DataFrame(globalBiasByWeek_stage2)

    # Save results
    print("\nSaving weekly analysis results...")
    noteParamsByWeek_stage1.to_csv(f"{output_dir}/note_params_weekly_stage1_{timestamp}.csv", index=False)
    raterParamsByWeek_stage1.to_csv(f"{output_dir}/rater_params_weekly_stage1_{timestamp}.csv", index=False)
    noteParamsByWeek_stage2.to_csv(f"{output_dir}/note_params_weekly_stage2_{timestamp}.csv", index=False)
    raterParamsByWeek_stage2.to_csv(f"{output_dir}/rater_params_weekly_stage2_{timestamp}.csv", index=False)
    globalBiasByWeek_stage1.to_csv(f"{output_dir}/global_bias_weekly_stage1_{timestamp}.csv", index=False)
    globalBiasByWeek_stage2.to_csv(f"{output_dir}/global_bias_weekly_stage2_{timestamp}.csv", index=False)
    
    print(f"Completed! Processed {len(weeks)} weeks.")
    return noteParamsByWeek_stage1, raterParamsByWeek_stage1, noteParamsByWeek_stage2, raterParamsByWeek_stage2, globalBiasByWeek_stage1, globalBiasByWeek_stage2


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run matrix factorization analysis')
    parser.add_argument('--output_dir', type=str, 
                        default='/accounts/projects/jchayes/commnotes/analysis/two_stage/output',
                        help='Directory to save output files')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("MATRIX FACTORIZATION BATCH ANALYSIS")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    try:
        # Run weekly analysis
        weekly_results = run_weekly_analysis(output_dir)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to: {output_dir}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nERROR: Analysis failed with exception:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
