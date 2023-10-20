# Entailment Classifier

## Singularity Image on SLURM

We must use
[Singularity](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/software/singularity-with-miniconda)
in order to first create a container with all of the dependencies which can then be used when submitting the 
SLURM batch job.

Note: make sure to use `overlay-50G-10M.ext3` or something with a higher capacity since JAX and Whisper seem to be quite heavy

Assuming the container is already set up (i.e. contains all of the dependencies per the above link):

```
cd /scratch/nn1331/whisper

# Following the above official documentation
srun --cpus-per-task=2 --mem=10GB --time=04:00:00 --pty /bin/bash

# wait to be assigned a node

singularity exec --overlay overlay-15GB-500K.ext3:rw /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash

source /ext3/env.sh
# activate the environment
```

This can be used for updating the dependencies within a Singularity Image as well!

Specifically, the following dependencies should be installed in the conatiner:
`pip3 install torch datasets evaluate numpy transformers scikit-learn hydra-core omegaconf bitarray sacrebleu`

### Dependencies

- [Whisper](https://github.com/sanchit-gandhi/whisper-jax#installation)

## On SLURM

```
# Submit job
sbatch whisper.sbatch

# Monitor job status
squeue -u $USER

# View job output
less whisper.out
```

### Copying data from Greene to local

Follow instructions found [here](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/hpc-storage/data-management/data-transfers)

One shot command: `scp nn1331@dtn.hpc.nyu.edu:/scratch/nn1331/whisper/data.csv .`

### Running jobs in interactive mode

```
srun --gres=gpu:1 --pty /bin/bash

# If you have the resources
srun --gres=gpu:a100:1 --mem-per-cpu=64GB --pty /bin/bash

# Within the session
singularity exec --nv --overlay /scratch/nn1331/whisper/whisper.ext3:ro \
	/scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
	/bin/bash

source /ext3/env.sh
```

### Setting custom `.cache`

```
export HF_HOME="/scratch/nn1331/whisper/.cache"
export TORCH_HOME="/scratch/nn1331/whisper/.cache"
export TRANSFORMERS_CACHE="/scratch/nn1331/whisper/.cache"
```

Additional
[resources](https://github.com/ZhaofengWu/lm_entailment#wills-notes-for-running-on-nyu-cluster)
to refer to

