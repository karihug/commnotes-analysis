#!/bin/bash
#SBATCH --job-name=mf-two-stage-final
#SBATCH --output=logs/mf_analysis_%j.log  # %j = job ID
#SBATCH --error=logs/mf_analysis_%j.err
#SBATCH --time=48:00:00                 # Max run time (hh:mm:ss) - increased for weekly processing
#SBATCH --mem=32G                       # Memory request - increased for large data
#SBATCH --cpus-per-task=8              # Number of CPU cores
#SBATCH --partition=high           # Or your cluster's partition name
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=krhuang@berkeley.edu

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="
echo ""

# Set output directory (can be overridden by command line argument)
OUTPUT_DIR="${1:-/accounts/projects/jchayes/commnotes/analysis/two_stage/output_final}"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the matrix factorization analysis
python run_mf.py --output_dir "$OUTPUT_DIR"

# Check exit status
EXIT_CODE=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
