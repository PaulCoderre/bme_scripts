#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=0-00:30:00
#SBATCH --mem=96G
#SBATCH --job-name=plots
#SBATCH --error=slurm_logs/slurm_%j.err
#SBATCH --output=slurm_logs/slurm_%j.out

# Add the directory where the modules are located to the MODULEPATH
. /work/comphyd_lab/local/modules/spack/2024v5/lmod-init-bash
module unuse $MODULEPATH
module use /work/comphyd_lab/local/modules/spack/2024v5/modules/linux-rocky8-x86_64/Core/

module restore scimods
source ~/virtual-envs/scienv/bin/activate


# Pass --top-only flag to only process rank 1
python -u 03_plot_skill_score.py







