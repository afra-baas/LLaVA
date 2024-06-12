# import wandb
# import yaml

# wandb.init()

# # Define the sweep configurations
# # fixed_batch_size_sweep = "/project/msc-thesis-project/forked_repos/LLaVA/scripts/v1_5/custom_eval/same_batch_size_sweep_copy.yaml"
# # optimal_batch_size_sweep = "optimal_batch_size_sweep.yaml"
# with open('/project/msc-thesis-project/forked_repos/LLaVA/scripts/v1_5/custom_eval/same_batch_size_sweep.yaml') as file:
#     fixed_batch_size_sweep = yaml.safe_load(file)
# print(fixed_batch_size_sweep)
# print(fixed_batch_size_sweep)

# # Initialize sweeps
# fixed_batch_sweep_id = wandb.sweep(fixed_batch_size_sweep, project="huggingface")
# # optimal_batch_sweep_id = wandb.sweep(optimal_batch_size_sweep, project="hyperparam_search_LLaVAs")

# # Define the training function
# def train():
#     import os
#     os.system("bash /project/msc-thesis-project/forked_repos/LLaVA/scripts/v1_5/custom_eval/finetune_lora_custom_copy.sh.sh")

# # Start the sweeps
# wandb.agent(fixed_batch_sweep_id, function=train, count=100)
# # wandb.agent(optimal_batch_sweep_id, function=train, count=100)


# # import wandb
# # import yaml

# # # Load the sweep configuration from the YAML file
# # with open("/project/msc-thesis-project/forked_repos/LLaVA/scripts/v1_5/custom_eval/same_batch_size_sweep_copy.yaml", "r") as file:
# #     sweep_config = yaml.safe_load(file)

# # print(sweep_config)
# # # Initialize the sweep
# # sweep_id = wandb.sweep(sweep_config)

# # # Define the training function
# # def train():
# #     import os
# #     os.system("bash finetune_lora_custom_copy.sh")

# # # Start the sweep
# # wandb.agent(sweep_id, function=train, count=100)




# # import wandb

# # # Define the sweep configurations
# # fixed_batch_size_sweep_config = {
# #     "name": "fixed-batch-size-sweep",
# #     "method": "bayes",
# #     "metric": {"name": "validation_accuracy", "goal": "maximize"},
# #     "project": "hyperparam_search_LLaVAs",
# #     "parameters": {
# #         "weight_decay": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-2},
# #         "lr": {"distribution": "log_uniform", "min": 2e-6, "max": 1e-6},
# #         "warmup_ratio": {"distribution": "uniform", "min": 0.01, "max": 0.1},
# #         "lr_scheduler_type": {"values": ["linear", "cosine"]},
# #         "mm_vision_select_layer": {"values": [-2, -1]},
# #         "mm_projector_lr": {"distribution": "log_uniform", "min": 2e-8, "max": 1e-4},
# #         "dataset": {"values": ["VSR"]},
# #         "method": {"values": ["no_depth"]}
# #     },
# #     "fixed": {
# #         "train_batchsize": 16,
# #         "eval_batchsize": 16
# #     },
# #     "run": {
# #         "count": 100
# #     }
# # }

# # # optimal_batch_size_sweep_config = {
# # #     "name": "optimal-batch-size-sweep",
# # #     "method": "bayes",
# # #     "metric": {"name": "validation_accuracy", "goal": "maximize"},
# # #     "project": "your_project_name",
# # #     "parameters": {
# # #         "epochs": {"values": [2, 3, 4, 5]},
# # #         "version": {"values": ["7b", "13b"]},
# # #         "device": {"values": [0, 1, 2, 3]},
# # #         "weight_decay": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-2},
# # #         "lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-3},
# # #         "warmup_ratio": {"distribution": "uniform", "min": 0.01, "max": 0.1},
# # #         "lr_scheduler_type": {"values": ["linear", "cosine"]},
# # #         "mm_vision_select_layer": {"values": [-2, -1, 0, 1, 2]},
# # #         "mm_projector_lr": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-4},
# # #         "train_batchsize": {"values": [8, 16, 32]},
# # #         "eval_batchsize": {"values": [8, 16, 32]},
# # #         "dataset": {"values": ["VSR", "VSR_BCE", "VSR_class2", "VSR_combi", "VSR_class", "Whatsup", "VSR_midas", "instruct_37k_VSR_class2_VSR_TF", "instruct_37k_VSR_class2", "Spatialvlm", "VSR_random", "VSR_random_BCE", "VSR_random_class"]},
# # #         "method": {"values": ["no_depth", "conv", "conv_hyper_param_lr5_nwd", "late", "resnet", "resnet_fusion", "mlp", "method2", "dino", "dino_late"]}
# # #     },
# # #     "run": {
# # #         "count": 200
# # #     }
# # # }

# # # Initialize sweeps
# # fixed_batch_sweep_id = wandb.sweep(fixed_batch_size_sweep_config, project="hyperparam_search_LLaVAs")
# # # optimal_batch_sweep_id = wandb.sweep(optimal_batch_size_sweep_config, project="your_project_name")

# # # Define the training function
# # def train():
# #     import os
# #     os.system("bash finetune_lora_custom_copy.sh")

# # # Start the sweeps
# # wandb.agent(fixed_batch_sweep_id, function=train, count=100)
# # # wandb.agent(optimal_batch_sweep_id, function=train, count=200)



# import os
# os.environ['WANDB_API_KEY'] = '6349ed9b917e6fee85da5729128e382f85ee0f53'

# import wandb
# wandb.login()

# # Define the sweep configurations
# sweep_config = {
#     'method': 'bayes',
#     'metric': {
#         'name': 'validation_accuracy',
#         'goal': 'maximize'
#     },
#     'parameters': {
#         'learning_rate': {
#             'distribution': 'log_uniform',
#             'min': 1e-6,
#             'max': 1e-4
#         },
#         'weight_decay': {
#             'distribution': 'log_uniform',
#             'min': 1e-6,
#             'max': 1e-4
#         },
#         'warmup_ratio': {
#             'distribution': 'uniform',
#             'min': 0.01,
#             'max': 0.1
#         },
#         'momentum': {
#             'distribution': 'uniform',
#             'min': 0.5,
#             'max': 0.99
#         },
#         'dropout_rate': {
#             'distribution': 'uniform',
#             'min': 0.0,
#             'max': 0.5
#         },
#         'lr_scheduler_type': {
#             'values': ['linear', 'cosine', 'polynomial', 'constant']
#         },
#         'mm_vision_select_layer': {
#             'values': [-2, -1]
#         },
#         'mm_projector_lr': {
#             'distribution': 'log_uniform',
#             'min': 1e-5,
#             'max': 1e-3
#         },
#         'dataset': {
#             'values': ['VSR']
#         },
#         'method': {
#             'values': ['no_depth']
#         }
#     }
# }



import os
os.environ['WANDB_API_KEY'] = '6349ed9b917e6fee85da5729128e382f85ee0f53'

import wandb
wandb.login()

# # Define the sweep configurations
# sweep_config = {
#     'method': 'bayes',
#     'metric': {
#         'name': 'validation_accuracy',
#         'goal': 'maximize'
#     },
#     'parameters': {
#         'learning_rate': {
#             'distribution': 'log_uniform',
#             'min': 1e-6,
#             'max': 1e-4
#         },
#         'weight_decay': {
#             'distribution': 'log_uniform',
#             'min': 1e-6,
#             'max': 1e-4
#         },
#         'warmup_ratio': {
#             'distribution': 'uniform',
#             'min': 0.01,
#             'max': 0.1
#         },
#         'momentum': {
#             'distribution': 'uniform',
#             'min': 0.5,
#             'max': 0.99
#         },
#         'dropout_rate': {
#             'distribution': 'uniform',
#             'min': 0.0,
#             'max': 0.5
#         },
#         'lr_scheduler_type': {
#             'values': ['linear', 'cosine', 'polynomial', 'constant']
#         },
#         'mm_vision_select_layer': {
#             'values': [-2, -1]
#         },
#         'mm_projector_lr': {
#             'distribution': 'log_uniform',
#             'min': 1e-5,
#             'max': 1e-3
#         },
#         'dataset': {
#             'values': ['VSR']
#         },
#         'method_': {
#             'values': ["no_depth" ]
#         }
#     }
# }


# sweep_config = {
#     'method': 'bayes',
#     'metric': {
#         'name': 'validation_accuracy',
#         'goal': 'maximize'
#     },
#     'parameters': {
#         'learning_rate': {
#             'distribution': 'log_uniform',
#             'log_uniform_values': [1e-6, 1e-4]  # Adjusted to original space limits
#         },
#         'weight_decay': {
#             'distribution': 'log_uniform',
#             'log_uniform_values': [1e-6, 1e-4]  # Adjusted to original space limits
#         },
#         'warmup_ratio': {
#             'distribution': 'uniform',
#             'min': 0.01,
#             'max': 0.1
#         },
#         'momentum': {
#             'distribution': 'uniform',
#             'min': 0.5,
#             'max': 0.99
#         },
#         'dropout_rate': {
#             'distribution': 'uniform',
#             'min': 0.0,
#             'max': 0.5
#         },
#         'lr_scheduler_type': {
#             'values': ['linear', 'cosine', 'polynomial', 'constant']
#         },
#         'mm_vision_select_layer': {
#             'values': [-2, -1]
#         },
#         'mm_projector_lr': {
#             'distribution': 'log_uniform',
#             'log_uniform_values': [1e-5, 1e-3]  # Adjusted to original space limits
#         },
#         'dataset': {
#             'values': ['VSR']
#         },
#         'method_': {
#             'values': ["no_depth"]
#         }
#     }
# }

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'validation_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        # 'lr': {
        #     'distribution': 'log_uniform',
        #     'min': -6,  # Exponent value for min (1e-6)
        #     'max': -4,  # Exponent value for max (1e-4)
        # },
        # 'weight_decay': {
        #     'distribution': 'log_uniform',
        #     'min': -6,  # Exponent value for min (1e-6)
        #     'max': -4,  # Exponent value for max (1e-4)
        # },
        # 'warmup_ratio': {
        #     'distribution': 'uniform',
        #     'min': 0.01,
        #     'max': 0.1
        # },
        # 'momentum': {
        #     'distribution': 'uniform',
        #     'min': 0.5,
        #     'max': 0.99
        # },
        # 'dropout_rate': {
        #     'distribution': 'uniform',
        #     'min': 0.0,
        #     'max': 0.5
        # },
        # 'lr_scheduler_type': {
        #     'values': ['linear', 'cosine', 'polynomial', 'constant']
        # },
        # 'mm_vision_select_layer': {
        #     'values': [-2, -1]
        # },
        'mm_projector_lr': {
            'distribution': 'log_uniform',
            'min': -5,  # Exponent value for min (1e-5)
            'max': -3,  # Exponent value for max (1e-3)
        },
        'dataset': {
            'values': ['VSR']
        },
        'method_': {
            'values': ["no_depth"]
        },
        'train_batchsize': {
            'values': [16]
        },
        'eval_batchsize': {
            'values': [16]
        }
    }
}


# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="hyperparam_search_LLaVAs")

# # Define the training function
# def train():
#     import os
#     os.system("bash /project/msc-thesis-project/forked_repos/LLaVA/scripts/v1_5/custom_eval/finetune_lora_custom_copy.sh")

def train():
    import os
    wandb.init()
    # lr = wandb.config.lr
    # weight_decay = wandb.config.weight_decay
    # warmup_ratio = wandb.config.warmup_ratio
    # lr_scheduler_type = wandb.config.lr_scheduler_type
    # mm_vision_select_layer = wandb.config.mm_vision_select_layer
    mm_projector_lr = wandb.config.mm_projector_lr
    # momentum = wandb.config.momentum
    # dropout_rate = wandb.config.dropout_rate
    dataset = wandb.config.dataset
    method_ = wandb.config.method_
    train_batchsize = wandb.config.train_batchsize
    eval_batchsize = wandb.config.eval_batchsize
    
    # Call the bash script with the specified hyperparameters
    # os.system(f"bash /project/msc-thesis-project/forked_repos/LLaVA/scripts/v1_5/custom_eval/finetune_lora_custom_copy.sh \
    #           {lr} {weight_decay} {warmup_ratio} {lr_scheduler_type} \
    #           {mm_vision_select_layer} {mm_projector_lr} {momentum} {dropout_rate} \
    #           {dataset} {method_} {train_batchsize} {eval_batchsize}")

    os.system(f"bash /project/msc-thesis-project/forked_repos/LLaVA/scripts/v1_5/custom_eval/finetune_lora_custom_copy.sh \
            {mm_projector_lr} \
            {dataset} {method_} {train_batchsize} {eval_batchsize}")




# Start the sweep
wandb.agent(sweep_id, function=train, count=100)




















