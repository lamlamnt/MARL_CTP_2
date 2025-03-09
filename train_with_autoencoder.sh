export XLA_PYTHON_CLIENT_PREALLOCATE=true 
export XLA_PYTHON_CLIENT_MEM_FRACTION=1

CUDA_VISIBLE_DEVICES=0 python train_autoencoder.py --n_node 10 --n_agent 2 --load_network_directory "node_10_agent_2" --graph_identifier "normalized_node_10_agent_2_prop_0.8" --log_directory "autoencoder_150" --autoencoder_weights_file "autoencoder_weights.flax" --prop_stoch 0.8 --latent_size 100
CUDA_VISIBLE_DEVICES=0 python main.py --n_node 10 --n_agent 2 --time_steps 4000000 --network_type "Densenet_Autoencoder" --log_directory "autoencoder_150" --graph_mode "load" --num_steps_before_update 2300 --graph_identifier "normalized_node_10_agent_2_prop_0.8" --prop_stoch 0.8 --num_stored_graphs 14000 --factor_inference_timesteps 2000 --clip_eps 0.115 --ent_coeff 0.08 --individual_reward_weight 0.28 --anneal_individual_reward_weight "linear" --num_critic_values 1 --reward_fail_to_service_goal_larger_index -0.4 --vf_coeff 0.18 --optimizer_norm_clip 2 --autoencoder_weights "autoencoder_150"

#Sweep for autoencoder:
#COPIED OVER BEST 10 nodes network
CUDA_VISIBLE_DEVICES=1 python train_autoencoder.py --n_node 10 --n_agent 2 --load_network_directory "best_10_nodes" --graph_identifier "normalized_node_10_agent_2_prop_0.8" --log_directory "sweep_autoencoder_training" --autoencoder_weights_file "autoencoder_weights.flax" --prop_stoch 0.8 --wandb_mode online --wandb_project_name sweep_autoencoder_training --wandb_sweep True --sweep_run_count 40 
