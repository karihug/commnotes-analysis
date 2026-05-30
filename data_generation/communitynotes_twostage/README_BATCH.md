# Running Matrix Factorization Analysis on SLURM

## Quick Start

### Default output directory
```bash
sbatch run_mf.sh
```

### Custom output directory
```bash
sbatch run_mf.sh /path/to/custom/output
```

## Files

- `run_mf.sh` - SLURM batch script
- `run_mf.py` - Python script for matrix factorization analysis
- `logs/` - Directory where SLURM logs will be saved

## Configuration

The SLURM script (`run_mf.sh`) is configured with:
- **Job name**: `mf-two-stage-final`
- **Memory**: 32GB
- **CPUs**: 8 cores
- **Time limit**: 48 hours
- **Partition**: high
- **Email notifications**: krhuang@berkeley.edu (on completion/failure)

## Output Files

The analysis will generate 6 CSV files in the output directory:
- `note_params_weekly_stage1_<timestamp>.csv` - Note parameters from Stage 1
- `rater_params_weekly_stage1_<timestamp>.csv` - Rater parameters from Stage 1
- `note_params_weekly_stage2_<timestamp>.csv` - Note parameters from Stage 2 (weighted)
- `rater_params_weekly_stage2_<timestamp>.csv` - Rater parameters from Stage 2 (weighted)
- `global_bias_weekly_stage1_<timestamp>.csv` - Global bias from Stage 1
- `global_bias_weekly_stage2_<timestamp>.csv` - Global bias from Stage 2

## Checking Job Status

```bash
# Check job status
squeue -u $USER

# View running job output
tail -f logs/mf_analysis_<job_id>.log

# Check for errors
tail -f logs/mf_analysis_<job_id>.err
```

## Canceling a Job

```bash
scancel <job_id>
```

## Manual Run (without SLURM)

```bash
python run_mf.py --output_dir /path/to/output
```

## Customizing the Analysis

Edit `run_mf.py` to modify:
- `max_date` - End date for data loading
- `start_date` - Start date for weekly analysis
- Data paths (RATINGS_DIR, etc.)
- Matrix factorization parameters (in `constants.py`)
