#!/bin/bash

seeds=(30 68)
declare -a probs=("Calcium_Oscillate")
declare -a methods=("smc")
n_ensemble=1000
##### Declare anything else needed to pass as an arg below

for prob in "${probs[@]}"; do
	for method in "${methods[@]}"; do
		for seed in "${seeds[@]}"; do
			echo "PE problem: $prob  Method: $method  RNG seed: $seed"
			outdir="./results/$prob/$method/"
			if [ ! -d "$outdir" ]; then
				mkdir -p $outdir
			fi

			logdir="$outdir/logs/"
			if [ ! -d "$logdir" ]; then
				mkdir -p $logdir
			fi
			
			job_name="$method.s$seed"
			out_file="$logdir${job_name}.out"
			err_file="$logdir${job_name}.err"

			command="sbatch --job-name=$job_name --output=$out_file --error=$err_file --export=method=$method,problem=$prob,seed=$seed,output_dir=$outdir,n_ensemble=$n_ensemble bigmem_slurm_job.sh"
			echo "$command"
			sbatch --job-name=$job_name --output=$out_file --error=$err_file --export=method=$method,problem=$prob,seed=$seed,output_dir=$outdir,n_ensemble=$n_ensemble bigmem_slurm_job.sh
			#bash single_slurm_job.sh

		done 
	done
done