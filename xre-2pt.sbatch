#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition rtx2080				### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 0-2:30:00					### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name xrayemb-rob		### name of the job
#SBATCH --output logs/xre-rob-2pt-%J.out	### output log for running job - %J for job number
#SBATCH --gpus=1							    ### number of GPUs, allocating more than 1 requires IT team's permission

#SBATCH --mail-user=uvp@cs.bgu.ac.il	        ### user's email for sending job status messages
#SBATCH --mail-type=END,FAIL					### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --mem=60G								### ammount of RAM memory, allocating more than 60G requires IT team's permission
#SBATCH --cpus-per-task=8				### number of CPU cores, allocating more than 10G requires IT team's permission

##SBATCH --tmp=10G
SLURM_JOB_NODELIST=cs-3090-02

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" ${SLURM_JOBID}
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ####
DATA_DIR=/home/pintery/git/xrayemb/corpus
MODEL_DIR=/home/pintery/git/xrayemb/models/roberta
TDT_MODEL_DIR=/home/pintery/git/xrayemb/models
LOG_DIR=/home/pintery/git/xrayemb/logs

module load anaconda				### load anaconda module (must be present when working with conda environments)
source activate yuval				### activate a conda environment, replace my_env with your conda environment

python -m src.tdt.run_tokdetok --seed 496351\
                                        --n-gpu 1\
										--model-type roberta\
										--model-dir ${MODEL_DIR}\
										--train-data-file ${DATA_DIR}/tweets-en-2016.text.gz\
										--eval-data-file ${DATA_DIR}/wikitext-2/wiki.test.tokens\
										--vocab-file ${DATA_DIR}/vocab\
										--output-dir ${TDT_MODEL_DIR}/${SLURM_JOBID}\
										--logdir ${LOG_DIR}/${SLURM_JOBID}\
										--block-size 512\
										--mlm-probability 0.15\
										--vocab-size 25000\
										--train-data-portion 0.001\
										--line-by-line\
										--shuffle-data\
										--lrn-tdt\
										--lrn-prob 0.1\
										--evaluate-during-training\
										--save-steps 5000\
										--save-total-limit 4\
										--learning-rate 2e-5\
										--num-train-epochs 1.0\
										--alpha-vec 0.1\
										--alpha-gen 0.075\
										--stochastic-inference\
										--char-emb-size 128\
										--train-cycle-dep\
										--cycle-freq 5000\
										--cycle-batch-iters 500\
										--td-strategy sqrt\
										--alpha-cyc-td 0.05\
										--alpha-cyc-dt 0.02\
										--vectorizer-type conv\
										--pool-policy max\
										--num-mlp-layers 2\
										--conv-hidden-size 256\
										--gen-lstm-hidden-size 256\
										--gen-num-lstm-layers 1\
										--spaces-end

										# --two-conv-proj
										# --checkpoint \
										# --lowercase-vocab\
										# --hashtml \
										# --generate-all \
