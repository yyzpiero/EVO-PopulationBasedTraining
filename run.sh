#! /bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh LOG_PATH RANK_NUM TRACK"
echo "For example: bash run.sh log 8 nasim:LargeGen-v0 True"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
module load mpi
LOG_PATH=$1
export LOG_PATH=${LOG_PATH}
RANK_NUM=$2
ENV_ID=$3 
TRACK=$4

now="$(date +"%I:%M")"
export now=${now}
agent_training_steps=20000
total_generations=50
EXEC_PATH=$(pwd)

if [[ ! -e ${EXEC_PATH}/${LOG_PATH}/${ENV_ID} ]]; then
    mkdir ${EXEC_PATH}/${LOG_PATH}/${ENV_ID}
fi

nohup mpirun -n ${RANK_NUM} --bind-to none python /home/yangyz/yangyz/codes/evo-clearn-ppo/pbt_rl_toy_trunt.py \
--env-id ${ENV_ID} --num-agents ${RANK_NUM} --track ${TRACK}  --total-generations ${total_generations} --use-sb True --agent-training-steps ${agent_training_steps} \
>${EXEC_PATH}/${LOG_PATH}/${ENV_ID}/num_agent_${RANK_NUM}_gen_${total_generations}_num_step_${agent_training_steps}_${now}.out 2>&1 &