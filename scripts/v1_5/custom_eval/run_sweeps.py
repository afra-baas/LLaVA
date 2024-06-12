import os
os.environ['WANDB_API_KEY'] = '6349ed9b917e6fee85da5729128e382f85ee0f53'
os.environ["WANDB_WATCH"] = "all"

import wandb
wandb.login()

sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'eval/loss',
        'goal': 'minimize'
    },
    'parameters': {
        'mm_projector_lr': {
            'distribution': 'log_uniform',
            'min': 1e-6,  # Exponent value for min (1e-5)
            'max': 1e-3,  # Exponent value for max (1e-3)
        },
        'lr': {
            'distribution': 'log_uniform',
            'min': 2e-5,  
            'max': 2e-3,  
        }
        # 'lr_scheduler_type': {
        #     'values': ['linear', 'cosine', 'polynomial', 'constant']
        # }
    }
}


# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="jupyter-proj")

def train():
    import os
    wandb.init(project="jupyter-proj")
    lr = wandb.config.lr
    # weight_decay = wandb.config.weight_decay
    # warmup_ratio = wandb.config.warmup_ratio
    # lr_scheduler_type = wandb.config.lr_scheduler_type
    # mm_vision_select_layer = wandb.config.mm_vision_select_layer
    mm_projector_lr = wandb.config.mm_projector_lr
    # momentum = wandb.config.momentum
    # dropout_rate = wandb.config.dropout_rate
    # dataset = wandb.config.dataset
    # method_ = wandb.config.method_
    # train_batchsize = wandb.config.train_batchsize
    # eval_batchsize = wandb.config.eval_batchsize
    
    # Call the bash script with the specified hyperparameters
    # os.system(f"bash /project/msc-thesis-project/forked_repos/LLaVA/scripts/v1_5/custom_eval/finetune_lora_custom_copy.sh \
    #           {lr} {weight_decay} {warmup_ratio} {lr_scheduler_type} \
    #           {mm_vision_select_layer} {mm_projector_lr} {momentum} {dropout_rate} \
    #           {dataset} {method_} {train_batchsize} {eval_batchsize}")

    os.system(f"bash /project/msc-thesis-project/forked_repos/LLaVA/scripts/v1_5/custom_eval/finetune_lora_custom_copy.sh \
            {mm_projector_lr} {lr}")


# Start the sweep
wandb.agent(sweep_id, function=train, count=50)




















