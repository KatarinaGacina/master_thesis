def get_config_train():
    config_params = {
        "batch_size": 16,
        "num_epochs": 40,
        "outputlen": 2000,
        "embed_dim": 16, 
        "embed_conv_kernel": 15, #must be odd
        "feature_dim": 128,
        "encoder_num": 2,
        "num_heads": 4,
        "pos_weight": 4.0,
        "negative_to_positive_ratio": 1, #not in use currently
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "wandb_name": "experiment_name",
        "wandb_project": "project_name"
    }

    return config_params

def get_config_train_gene():
    config_params = {
        "batch_size": 1,
        "num_epochs": 40,
        "outputlen": 99968,
        "embed_dim": 16,
        "embed_conv_kernel": 15, #must be odd
        "feature_dim": 128,
        "encoder_num": 2,
        "num_heads": 4,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "pos_weight": 3.0,
        "wandb_name": "experiment_name",
        "wandb_project": "project_name"
    }

    return config_params

def get_config_base():
    config_params = {
        "batch_size": 256,
        "num_epochs": 30,
        "specified_len": 2000,
        "embed_dim": 16,
        "embed_conv_kernel": 15, #must be odd
        "feature_dim": 128,
        "encoder_num": 2,
        "num_heads": 4,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "output_vocab_size": 4,
        "weight": None,
        "wandb_name": "experiment_name",
        "wandb_project": "project_name"
    }

    return config_params