#!/bin/bash
#SBATCH --clusters=htc
#SBATCH --nodelist=htc-g053
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=short
#SBATCH --job-name=sweep-10-nodes-2-critics
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
srun python main.py --n_node 10 --n_agent 2 --log_directory "sweep_2_critic_10_nodes" --time_steps 4000000 --network_type "Densenet_Same" --graph_mode "load" --learning_rate 0.001 --num_steps_before_update 2300 --individual_reward_weight 1 --anneal_individual_reward_weight "linear" --num_critic_values 2 --clip_eps 0.10255 --num_update_epochs 6 --ent_coeff_schedule "sigmoid" --vf_coeff 0.135 --ent_coeff 0.11246 --deterministic_inference_policy True --graph_identifier "normalized_node_10_agent_2_prop_0.8" --prop_stoch 0.8 --num_stored_graphs 14000 --factor_inference_timesteps 2000 --num_minibatches 1 --optimizer_norm_clip 2.0 --frequency_testing 20 --factor_testing_timesteps 50 --wandb_mode online --wandb_project_name sweep_2_critic_10_nodes --yaml_file "sweep_2_critic.yaml" --wandb_sweep True --sweep_run_count 10 --reward_fail_to_service_goal_larger_index -0.61 --reward_exceed_horizon -4.7

rsync -av --exclude=input --exclude=bin ./Logs/sweep_2_critic_10_nodes $DATA/MARL_CTP_2/ARC