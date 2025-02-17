#Generate graphs for 1 agent
CUDA_VISIBLE_DEVICES=1 python main.py --n_node 10 --n_agent 1 --time_steps 100000 --network_type "Densenet_Same" --log_directory "store" --graph_mode "store" --num_steps_before_update 100 --densenet_num_layers "2,2,2" --densenet_bn_size 2 --densenet_growth_rate 10 --frequency_testing 10 --graph_identifier "node_10_agent_1_prop_0.4" --prop_stoch 0.4 --num_stored_graphs 14000 --factor_inference_timesteps 1000

#Run for 1 agent individual reward - expect to see roughly same performance as single agent code
CUDA_VISIBLE_DEVICES=1 python main.py --n_node 10 --n_agent 1 --time_steps 3000000 --network_type "Densenet_Same" --log_directory "generalize_individual_10_nodes_1_agent" --graph_mode "load" --num_steps_before_update 1800 --graph_identifier "node_10_agent_1_prop_0.4" --prop_stoch 0.4 --num_stored_graphs 14000 --factor_inference_timesteps 1000 --clip_eps 0.1 --individual_reward_weight 1 

#Generate graphs for 0.8 - 2 agents
CUDA_VISIBLE_DEVICES=1 python main.py --n_node 10 --n_agent 2 --time_steps 100000 --network_type "Densenet_Same" --log_directory "store" --graph_mode "store" --num_steps_before_update 100 --densenet_num_layers "2,2,2" --densenet_bn_size 2 --densenet_growth_rate 10 --frequency_testing 10 --graph_identifier "node_10_agent_2_prop_0.8" --prop_stoch 0.8 --num_stored_graphs 14000 --factor_inference_timesteps 1000

#Test for team reward - 2 agents prop 0.4
CUDA_VISIBLE_DEVICES=1 python main.py --n_node 10 --n_agent 2 --time_steps 3000000 --network_type "Densenet_Same" --log_directory "generalize_team_10_nodes_2_agents" --graph_mode "load" --num_steps_before_update 1800 --graph_identifier "node_10_agent_2" --num_stored_graphs 14000 --factor_inference_timesteps 1000 --clip_eps 0.1 --individual_reward_weight 0

#Test 0.5 combination - 2 agents prop 0.4
CUDA_VISIBLE_DEVICES=1 python main.py --n_node 10 --n_agent 2 --time_steps 3000000 --network_type "Densenet_Same" --log_directory "generalize_0.5_10_nodes_2_agents" --graph_mode "load" --num_steps_before_update 1800 --graph_identifier "node_10_agent_2" --num_stored_graphs 14000 --factor_inference_timesteps 1000 --clip_eps 0.1 --individual_reward_weight 0.5
