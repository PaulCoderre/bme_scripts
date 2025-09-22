#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --time=0-1:00:00
#SBATCH --mem=96G
#SBATCH --job-name=hydrobm
#SBATCH --error=slurm_logs/hydrobm.err
#SBATCH --output=slurm_logs/hydrobm.out

# Add the directory where the modules are located to the MODULEPATH
. /work/comphyd_lab/local/modules/spack/2024v5/lmod-init-bash
module unuse $MODULEPATH
module use /work/comphyd_lab/local/modules/spack/2024v5/modules/linux-rocky8-x86_64/Core/

module restore scimods
source ~/virtual-envs/scienv/bin/activate


python -u run_hydrobm_skill_score.py





