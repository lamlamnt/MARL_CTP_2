#!/bin/bash
#SBATCH --clusters=htc
#SBATCH --nodelist=htc-g053
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --job-name=10_nodes_2_agents
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

#Cmds to run to get learning curve + prop 0.4 data
#1 critic - individual - constant
srun python main.py --n_node 10 --n_agent 2 --time_steps 4000000 --network_type "Densenet_Same" --log_directory "experiment_10_nodes_2_agents_1_critic_individual" --graph_mode "load" --num_steps_before_update 2300 --graph_identifier "normalized_node_10_agent_2_prop_0.8" --prop_stoch 0.8 --num_stored_graphs 14000 --factor_inference_timesteps 2000 --individual_reward_weight 1 --anneal_individual_reward_weight "constant" --num_critic_values 1 --clip_eps 0.10255 --ent_coeff 0.11246 --vf_coeff 0.135 --reward_fail_to_service_goal_larger_index -0.61 --reward_exceed_horizon -4.7 --factor_testing_timesteps 200 --num_update_epochs 6 --deterministic_inference_policy True --optimizer_norm_clip 2.0 --learning_rate 0.001
rsync -av --exclude=input --exclude=bin ./Logs/experiment_10_nodes_2_agents_1_critic_individual $DATA/MARL_CTP_2/ARC

#1 critic - team - constant
srun python main.py --n_node 10 --n_agent 2 --time_steps 4000000 --network_type "Densenet_Same" --log_directory "experiment_10_nodes_2_agents_1_critic_team" --graph_mode "load" --num_steps_before_update 2300 --graph_identifier "normalized_node_10_agent_2_prop_0.8" --prop_stoch 0.8 --num_stored_graphs 14000 --factor_inference_timesteps 2000 --individual_reward_weight 0 --anneal_individual_reward_weight "constant" --num_critic_values 1 --clip_eps 0.10255 --ent_coeff 0.11246 --vf_coeff 0.135 --reward_fail_to_service_goal_larger_index -0.61 --reward_exceed_horizon -4.7 --factor_testing_timesteps 200 --num_update_epochs 6 --deterministic_inference_policy True --optimizer_norm_clip 2.0 --learning_rate 0.001
rsync -av --exclude=input --exclude=bin ./Logs/experiment_10_nodes_2_agents_1_critic_team $DATA/MARL_CTP_2/ARC

#1 critic - mix at 0.5 - no decay
srun python main.py --n_node 10 --n_agent 2 --time_steps 4000000 --network_type "Densenet_Same" --log_directory "experiment_10_nodes_2_agents_1_critic_mixed_no_decay" --graph_mode "load" --num_steps_before_update 2300 --graph_identifier "normalized_node_10_agent_2_prop_0.8" --prop_stoch 0.8 --num_stored_graphs 14000 --factor_inference_timesteps 2000 --individual_reward_weight 0.5 --anneal_individual_reward_weight "constant" --num_critic_values 1 --clip_eps 0.10255 --ent_coeff 0.11246 --vf_coeff 0.135 --reward_fail_to_service_goal_larger_index -0.61 --reward_exceed_horizon -4.7 --factor_testing_timesteps 200 --num_update_epochs 6 --deterministic_inference_policy True --optimizer_norm_clip 2.0 --learning_rate 0.001
rsync -av --exclude=input --exclude=bin ./Logs/experiment_10_nodes_2_agents_1_critic_mixed_no_decay $DATA/MARL_CTP_2/ARC

#2 critic linear decay starting from 0.5
srun python main.py --n_node 10 --n_agent 2 --time_steps 4000000 --network_type "Densenet_Same" --log_directory "experiment_10_nodes_2_agents_2_critic_decay" --graph_mode "load" --num_steps_before_update 2300 --graph_identifier "normalized_node_10_agent_2_prop_0.8" --prop_stoch 0.8 --num_stored_graphs 14000 --factor_inference_timesteps 2000 --individual_reward_weight 0.3767 --anneal_individual_reward_weight "linear" --num_critic_values 2 --clip_eps 0.10255 --ent_coeff 0.11246 --vf_coeff 0.135 --reward_fail_to_service_goal_larger_index -0.61 --reward_exceed_horizon -4.7 --factor_testing_timesteps 200 --num_update_epochs 6 --deterministic_inference_policy True --optimizer_norm_clip 2.0 --learning_rate 0.001
rsync -av --exclude=input --exclude=bin ./Logs/experiment_10_nodes_2_agents_2_critic_decay $DATA/MARL_CTP_2/ARC

#1 critic best 
srun python main.py --n_node 10 --n_agent 2 --time_steps 4000000 --network_type "Densenet_Same" --log_directory "experiment_10_nodes_2_agents_1_critic_decay_best" --graph_mode "load" --num_steps_before_update 2300 --graph_identifier "normalized_node_10_agent_2_prop_0.8" --prop_stoch 0.8 --num_stored_graphs 14000 --factor_inference_timesteps 2000 --individual_reward_weight 0.3767 --anneal_individual_reward_weight "linear" --num_critic_values 1 --clip_eps 0.10255 --ent_coeff 0.11246 --vf_coeff 0.135 --reward_fail_to_service_goal_larger_index -0.61 --reward_exceed_horizon -4.7 --factor_testing_timesteps 200 --num_update_epochs 6 --deterministic_inference_policy True --optimizer_norm_clip 2.0 --learning_rate 0.001
rsync -av --exclude=input --exclude=bin ./Logs/experiment_10_nodes_2_agents_1_critic_decay_best $DATA/MARL_CTP_2/ARC

#1 critic - best prop 0.4 
srun python main.py --n_node 10 --n_agent 2 --time_steps 4000000 --network_type "Densenet_Same" --log_directory "experiment_10_node_2_agents_prop_0.4_best" --graph_mode "load" --num_steps_before_update 2300 --graph_identifier "normalized_node_10_agent_2_prop_0.4" --prop_stoch 0.4 --num_stored_graphs 14000 --factor_inference_timesteps 2000 --individual_reward_weight 0.3767 --anneal_individual_reward_weight "linear" --num_critic_values 1 --clip_eps 0.10255 --ent_coeff 0.11246 --vf_coeff 0.135 --reward_fail_to_service_goal_larger_index -0.61 --reward_exceed_horizon -4.7 --factor_testing_timesteps 200 --num_update_epochs 6 --deterministic_inference_policy True --optimizer_norm_clip 2.0 --learning_rate 0.001
rsync -av --exclude=input --exclude=bin ./Logs/experiment_10_node_2_agents_prop_0.4_best $DATA/MARL_CTP_2/ARC

