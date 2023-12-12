echo "$@"
. /home/miniconda3/etc/profile.d/conda.sh
conda activate cubes
export LD_LIBRARY_PATH=/home/miniconda3/envs/rl3sb3/lib/:$LD_LIBRARY_PATH
cd /global/scratch/users/oleh/taskmaster_sb3
python train_her.py "$@"