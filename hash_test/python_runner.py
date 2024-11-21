from run import main

import yaml
import wandb

# Add requirement for wandb core
wandb.require("core")

# Define the path to your YAML file
# yaml_file_path = '../configs/qm9/alpha.yaml'
# yaml_file_path = '../configs/qm9/alpha_pos.yaml'
# yaml_file_path = '../configs/qm9/ppgn.yaml'
# yaml_file_path = '../configs/qm9/alpha_pos_test.yaml'
# yaml_file_path = '../configs/qm9/alpha_pos_downstream.yaml'


# yaml_file_path = '../configs/zinc/best_1l_metric.yaml'
yaml_file_path = '../configs/zinc/best_1l_metric_ppgn.yaml'

# Load the YAML file
with open(yaml_file_path, 'r') as file:
    config = yaml.safe_load(file)

config['debug'] = True

# Print the dictionary to verify
print(config)

main(config['fixed'])