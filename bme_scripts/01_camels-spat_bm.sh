#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0-1:00:00
#SBATCH --mem=96G
#SBATCH --job-name=scenarios
#SBATCH --error=slurm_logs/slurm_%j.err
#SBATCH --output=slurm_logs/slurm_%j.out
#SBATCH --array=1-40

# Add the directory where the modules are located to the MODULEPATH
. /work/comphyd_lab/local/modules/spack/2024v5/lmod-init-bash
module unuse $MODULEPATH
module use /work/comphyd_lab/local/modules/spack/2024v5/modules/linux-rocky8-x86_64/Core/

module restore scimods
source ~/virtual-envs/scienv/bin/activate

run_number=$SLURM_ARRAY_TASK_ID

python -u 01_camels-spat_bm.py $run_number





