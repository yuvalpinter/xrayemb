#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition rtx2080					### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 0-2:30:00					### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name xrayemb-roberta-wnut		### name of the job
#SBATCH --output logs/xre-rob-wnut-%J.out	### output log for running job - %J for job number
#SBATCH --gpus=1							### number of GPUs, allocating more than 1 requires IT team's permission

#SBATCH --mail-user=uvp@cs.bgu.ac.il	        ### user's email for sending job status messages
#SBATCH --mail-type=END,FAIL					### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --mem=6G								### ammount of RAM memory, allocating more than 60G requires IT team's permission
#SBATCH --cpus-per-task=4				### number of CPU cores, allocating more than 10G requires IT team's permission

##SBATCH --tmp=10G
SLURM_JOB_NODELIST=cs-3090-02

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ####
DATA_DIR=/home/pintery/git/xrayemb/datasets/wnut16
MODEL_DIR=/home/pintery/git/xrayemb/models/roberta
TDT_MODEL_DIR=/home/pintery/git/xrayemb/models/202103192237
OUTPUT_DIR=/home/pintery/git/xrayemb/output/wnut

module load anaconda				### load anaconda module (must be present when working with conda environments)
source activate yuval				### activate a conda environment, replace my_env with your conda environment

python -m src.downstream.run_downstream --n-gpu 1\
										--model-type roberta\
										--base-model-dir ${MODEL_DIR}\
										--out-dir ${OUTPUT_DIR}\
										--tdt-model-dir ${TDT_MODEL_DIR}\
										--vectorizer-type conv\
										--infer-policy stoch\
										--stoch-rate 0.0001\
										--pool-policy max\
										--finetune\
										--ft-lr 2e-5\
										--alpha-tdt 0.0\
										--alpha-vec 0.1\
										--alpha-gen 0.075\
										--task-lr 5e-4\
										--task-warmup 0.1\
										--batch-size 2\
										--dataset ${DATA_DIR}\
										--data-sample 1.0\
										--stopping-patience 4\
										--epochs 20
										
										# --second-base-model-dir \
										# --evaluate-file\
										# --checkpoint \
										# --logdir \
										# --lowercase-vocab\
										# --lowercase\
										# --ft-wd \
										# --ft-adam-eps \
										# --hashtml \
										# --task-lstm-hidden-size \
										# --task-num-lstm-layers \
										# --task-num-mlp-layers \
										# --margin-loss\
