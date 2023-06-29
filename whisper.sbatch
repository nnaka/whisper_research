#!/bin/bash 
#SBATCH --nodes=3                        # requests 3 compute servers
#SBATCH --ntasks-per-node=2              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=1:00:00
#SBATCH --mem=2GB
#SBATCH --job-name=python-whisper
#SBATCH --output=python-whisper.out

module purge
module load python/intel/3.8.6

python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
./whisper.py