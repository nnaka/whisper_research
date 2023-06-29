# Entailment Classifier

## Installation

```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Run

```
./entailment_classifier.py
```

### On SLURM

```
# Submit job
sbatch entailment_classifier.sbatch

# Monitor job status
squeue -u $USER

# View job output
cat python-entailment-classifier.out
```

## Conda

```
module load anaconda3/2020.07
conda env create -f environment.yml
conda activate entailment
```
