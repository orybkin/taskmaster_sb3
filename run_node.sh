#! /bin/bash

echo $SLURM_JOB_ID
echo slurm job id
singularity run -B /var/lib/dcv-gl --nv --writable-tmpfs  --bind /global/scratch/users/oleh/tmp:/tmp --bind /global/scratch/users/oleh/miniconda3/:/home/miniconda3 /global/scratch/users/oleh/taskmaster_sb3.sif -- bash /global/scratch/users/oleh/taskmaster_sb3/run_container.sh  --slurm_job_id=$SLURM_JOB_ID "$@"