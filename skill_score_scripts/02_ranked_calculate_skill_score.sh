#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-00:30:00
#SBATCH --mem=96G
#SBATCH --job-name=scenarios
#SBATCH --error=slurm_logs/slurm_%j.err
#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --array=1-39

# Add the directory where the modules are located to the MODULEPATH
. /work/comphyd_lab/local/modules/spack/2024v5/lmod-init-bash
module unuse $MODULEPATH
module use /work/comphyd_lab/local/modules/spack/2024v5/modules/linux-rocky8-x86_64/Core/

module restore scimods
source ~/virtual-envs/scienv/bin/activate

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Array Task Count: $SLURM_ARRAY_TASK_COUNT"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "=========================================="

Rscript 02_calculate_skill_score.R





