#!/bin/bash
#SBATCH --job-name=TEST5_NPF_PROJ
#SBATCH --time=05:00:00
#SBATCH --output=./logs/%x_%a_%A.out
#SBATCH --error=./logs/%x_%a_%A.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=14
#SBATCH --cpus-per-task=2
#SBATCH --mail-user=joao.reboucas@unesp.br
#SBATCH --mail-type=ALL

RUNTYPE=no_pairfix_projected # Control whether to run phase a, phase b, no_pairfix or no_pairfix_projected
INPUT=/home/joaoreboucas/COLA_projects/test_5/lua_files_${RUNTYPE}/parameter_file${SLURM_ARRAY_TASK_ID}.lua

echo "Job started in `hostname`"

conda activate cola
source start_cola

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PROC_BIND=close

mpirun -n ${SLURM_NTASKS_PER_NODE} --mca btl vader,tcp,self --bind-to core --rank-by core --map-by core ./FML/COLASolver/nbody ${INPUT}
