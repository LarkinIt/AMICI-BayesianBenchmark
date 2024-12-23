#!/bin/bash
#SBATCH -N 1
#SBATCH -p any_cpu
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --mail-user=caroline.larkin@pitt.edu

export PATH=/net/dali/home/mscbio/cil8/.local/bin:$PATH
export CPPFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

module unload anaconda
module load anaconda/3-cluster

eval "$(conda shell.bash hook)"
conda activate bayes
conda install -c conda-forge ncurses
python run_model_calibration.py -m ${method} -p ${problem} -n ${n_ensemble} -s ${seed} -o ${output_dir} -c ${SLURM_CPUS_PER_TASK}

# Leave this line to tell slurm that the script finished correctly
exit 0