#! /bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh LOG_PATH RANK_NUM TRACK"
echo "For example: bash run.sh log 8 nasim:LargeGen-v0 True"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

source ~/software/miniconda3/etc/profile.d/conda.sh
conda activate poetry-mpi
#module load mpi
LOG_PATH=$1
export LOG_PATH=${LOG_PATH}
RANK_NUM=$2
ENV_ID=$3 
TRACK=$4

now="$(date +"%I:%M")"
export now=${now}
agent_training_steps=50000
total_generations=20
EXEC_PATH=$(pwd)

if [[ ! -e ${EXEC_PATH}/${LOG_PATH}/${ENV_ID} ]]; then
    mkdir ${EXEC_PATH}/${LOG_PATH}/${ENV_ID}
fi


nohup mpirun -n ${RANK_NUM} --bind-to none python ./pbt_rl_toy_trunt.py \
--env-id ${ENV_ID} --num-agents ${RANK_NUM} --track ${TRACK} --num-envs 8 --total-generations ${total_generations} --use-sb False --agent-training-steps ${agent_training_steps} \
>${EXEC_PATH}/${LOG_PATH}/${ENV_ID}/EVO_num_agent_${RANK_NUM}_gen_${total_generations}_num_step_${agent_training_steps}_${now}.out 2>&1 & 

wait 

nohup mpirun -n ${RANK_NUM} --bind-to none python ./pbt_rl_toy_trunt.py \
--env-id ${ENV_ID} --num-agents ${RANK_NUM} --track ${TRACK} --num-envs 8 --total-generations ${total_generations} --use-sb True --agent-training-steps ${agent_training_steps} \
>${EXEC_PATH}/${LOG_PATH}/${ENV_ID}/SB_num_agent_${RANK_NUM}_gen_${total_generations}_num_step_${agent_training_steps}_${now}.out 2>&1 &


