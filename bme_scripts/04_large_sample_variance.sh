#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0-00:20:00
#SBATCH --mem=96G
#SBATCH --job-name=scenarios
#SBATCH --error=slurm_logs/slurm_%j.err
#SBATCH --output=slurm_logs/slurm_%j.out

# Add the directory where the modules are located to the MODULEPATH
. /work/comphyd_lab/local/modules/spack/2024v5/lmod-init-bash
module unuse $MODULEPATH
module use /work/comphyd_lab/local/modules/spack/2024v5/modules/linux-rocky8-x86_64/Core/

module restore scimods
source ~/virtual-envs/scienv/bin/activate


python 04_cs_large_sample_variance_components.py





