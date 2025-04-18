#!/bin/bash
#SBATCH --clusters=htc
#SBATCH --nodelist=htc-g053
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --job-name=20nodes2agents
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lam.lam@pmb.ox.ac.uk

cd $SCRATCH || exit 1

rsync -av $DATA/MARL_CTP_2 ./

module purge
module load Anaconda3
source activate /data/engs-goals/pemb6454/marl_3

echo "Copied MARL folder and entered conda environment"
export XLA_PYTHON_CLIENT_PREALLOCATE=true 
export XLA_PYTHON_CLIENT_MEM_FRACTION=1
cd MARL_CTP_2

export WANDB_API_KEY="267ce0358c86cf2f418a382f7be9e01c6464c124"
srun python main.py --n_node 30 --n_agent 1 --log_directory "normalized_30_nodes_1_agent" --time_steps 10000000 --network_type "Densenet_Same" --graph_mode "load" --learning_rate 0.001 --num_steps_before_update 3000 --individual_reward_weight 1 --anneal_individual_reward_weight "linear" --num_critic_values 1 --clip_eps 0.1 --num_update_epochs 6 --ent_coeff_schedule "sigmoid" --vf_coeff 0.128 --ent_coeff 0.174 --deterministic_inference_policy True --graph_identifier "node_20_agent_2_prop_0.8" --prop_stoch 0.8 --num_stored_graphs 30000 --factor_inference_timesteps 2000 --num_minibatches 1 --optimizer_norm_clip 1.0 --frequency_testing 20 --factor_testing_timesteps 50 --densenet_bn_size 4 --densenet_growth_rate 32 --wandb_mode online --wandb_project_name initial_node20_2_agents

rsync -av --exclude=input --exclude=bin ./Logs/normalized_30_nodes_1_agent $DATA/MARL_CTP_2/ARC