#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=lr7
#SBATCH --account=pc_qmmgnn
#SBATCH --time=72:00:00
#SBATCH --job-name="XGBoost_tiles_no_yeast_CV"
#SBATCH --output="XGBoost_tiles_no_yeast_CV.out"
#SBATCH --error="XGBoost_tiles_no_yeast_CV.err"
#SBATCH --comment "Run notebook XGBoost_tiles_no_yeast_CV"
#SBATCH --qos=lr_normal

#SBATCH --cpus-per-task=56
#SBATCH --mem=128G

echo "Running notebook"

python notebooks/XGBoost_tiles_no_yeast_CV.py

echo "Done"

