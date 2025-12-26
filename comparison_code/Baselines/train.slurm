#!/bin/bash
#SBATCH --nodes=1 #total number of nodes for the job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB # Requests 48GB of memory per node
#SBATCH --ntasks-per-node=1 #how many copies of code to run
#SBATCH --time=24:00:00 #amount of time for the whole job
#SBATCH --partition=gpu #the queue/partition to run on
#SBATCH --gres=gpu:1
#SBATCH --account=ivy-hip-rderi
#SBATCH --output=./slurm_output/slurm.out
#SBATCH --error=./slurm_output/slurm.err

module purge
module load miniforge
source activate ood_env
cd /standard/ivy-hip-rderi/ss4yd/DETRtime/Baselines
python main.py

