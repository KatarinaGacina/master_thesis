def get_config_finetune():
    config_params = {
        "model_type": "standard",
        "embed_dim": 16, 
        "embed_conv_kernel": 15, #must be odd
        "feature_dim": 128,
        "encoder_num": 2,
        "num_heads": 4,
        "pretrained": None,
        "num_epochs": 20,
        "batch_size": 128,
        "eval_batch_size": 128,
        "learning_rate": 1e-5,
        "logging_steps": 50,
        "output_path": "/path"
    }

    return config_params

def get_config_task(task):
    if task == "promoter_all":
        return {
            "task": "promoter_all",
            "number_labels": 2,
            "outputlen": 300,
            "metric": "f1",
        }
    elif task == "promoter_tata":
        return {
            "task": "promoter_tata",
            "number_labels": 2,
            "outputlen": 300,
            "metric": "f1",
        }
    elif task == "promoter_no_tata":
        return {
            "task": "promoter_no_tata",
            "number_labels": 2,
            "outputlen": 300,
            "metric": "f1",
        }
    elif task == "enhancers":
        return {
            "task": "enhancers",
            "number_labels": 2,
            "outputlen": 200,
            "metric": "mcc",
        }
    elif task == "enhancers_types":
        return {
            "task": "enhancers_types",
            "number_labels": 3,
            "outputlen": 200,
            "metric": "mcc",
        }
    elif task == "H3":
        return {
            "task": "H3",
            "number_labels": 2,
            "outputlen": 500,
            "metric": "mcc",
        }
    elif task == "H3K4me1":
        return {
            "task": "H3K4me1",
            "number_labels": 2,
            "outputlen": 500,
            "metric": "mcc",
        }
    elif task == "H3K4me2":
        return {
            "task": "H3K4me2",
            "number_labels": 2,
            "outputlen": 500,
            "metric": "mcc",
        }
    elif task == "H3K4me3":
        return {
            "task": "H3K4me3",
            "number_labels": 2,
            "outputlen": 500,
            "metric": "mcc",
        }
    elif task == "H3K9ac":
        return {
            "task": "H3K9ac",
            "number_labels": 2,
            "outputlen": 500,
            "metric": "mcc",
        }
    elif task == "H3K14ac":
        return {
            "task": "H3K14ac",
            "number_labels": 2,
            "outputlen": 500,
            "metric": "mcc",
        }
    elif task == "H3K36me3":
        return {
            "task": "H3K36me3",
            "number_labels": 2,
            "outputlen": 500,
            "metric": "mcc",
        }
    elif task == "H3K79me3":
        return {
            "task": "H3K79me3",
            "number_labels": 2,
            "outputlen": 500,
            "metric": "mcc",
        }
    elif task == "H4":
        return {
            "task": "H4",
            "number_labels": 2,
            "outputlen": 500,
            "metric": "mcc",
        }
    elif task == "H4ac":
        return {
            "task": "H4ac",
            "number_labels": 2,
            "outputlen": 500,
            "metric": "mcc",
        }
    elif task == "splice_sites_acceptors":
        return {
            "task": "splice_sites_acceptors",
            "number_labels": 2,
            "outputlen": 600,
            "metric": "f1",
        }
    elif task == "splice_sites_all":
        return {
            "task": "splice_sites_all",
            "number_labels": 3,
            "outputlen": 400,
            "metric": "acc",
        }
    elif task == "splice_sites_donors":
        return {
            "task": "splice_sites_donors",
            "number_labels": 2,
            "outputlen": 600,
            "metric": "f1",
        }


#for InstaDeepAI/nucleotide_transformer_downstream_tasks_revised
def get_config_revised_task(task):
    if task == "promoter_all":
        return {
            "task": "promoter_all",
            "number_labels": 2,
            "outputlen": 300,
            "metric": "f1",
        }
    elif task == "promoter_tata":
        return {
            "task": "promoter_tata",
            "number_labels": 2,
            "outputlen": 300,
            "metric": "f1",
        }
    elif task == "promoter_no_tata":
        return {
            "task": "promoter_no_tata",
            "number_labels": 2,
            "outputlen": 300,
            "metric": "f1",
        }
    elif task == "enhancers":
        return {
            "task": "enhancers",
            "number_labels": 2,
            "outputlen": 400,
            "metric": "mcc",
        }
    elif task == "enhancers_types":
        return {
            "task": "enhancers_types",
            "number_labels": 3,
            "outputlen": 400,
            "metric": "mcc",
        }
    elif task == "H2AFZ":
        return {
            "task": "H2AFZ",
            "number_labels": 2,
            "outputlen": 1000,
            "metric": "mcc",
        }
    elif task == "H3K4me1":
        return {
            "task": "H3K4me1",
            "number_labels": 2,
            "outputlen": 1000,
            "metric": "mcc",
        }
    elif task == "H3K4me2":
        return {
            "task": "H3K4me2",
            "number_labels": 2,
            "outputlen": 1000,
            "metric": "mcc",
        }
    elif task == "H3K4me3":
        return {
            "task": "H3K4me3",
            "number_labels": 2,
            "outputlen": 1000,
            "metric": "mcc",
        }
    elif task == "H3K9ac":
        return {
            "task": "H3K9ac",
            "number_labels": 2,
            "outputlen": 1000,
            "metric": "mcc",
        }
    elif task == "H3K27me3":
        return {
            "task": "H3K27me3",
            "number_labels": 2,
            "outputlen": 1000,
            "metric": "mcc",
        }
    elif task == "H3K36me3":
        return {
            "task": "H3K36me3",
            "number_labels": 2,
            "outputlen": 1000,
            "metric": "mcc",
        }
    elif task == "H4K20me1":
        return {
            "task": "H4K20me1",
            "number_labels": 2,
            "outputlen": 1000,
            "metric": "mcc",
        }
    elif task == "H3K27ac":
        return {
            "task": "H3K27ac",
            "number_labels": 2,
            "outputlen": 1000,
            "metric": "mcc",
        }
    elif task == "H3K9me3":
        return {
            "task": "H3K9me3",
            "number_labels": 2,
            "outputlen": 1000,
            "metric": "mcc",
        }
    elif task == "splice_sites_acceptors":
        return {
            "task": "splice_sites_acceptors",
            "number_labels": 2,
            "outputlen": 600,
            "metric": "f1",
        }
    elif task == "splice_sites_all":
        return {
            "task": "splice_sites_all",
            "number_labels": 3,
            "outputlen": 600,
            "metric": "acc",
        }
    elif task == "splice_sites_donors":
        return {
            "task": "splice_sites_donors",
            "number_labels": 2,
            "outputlen": 600,
            "metric": "f1",
        }
