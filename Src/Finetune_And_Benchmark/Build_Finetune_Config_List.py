import sys
import math
import torch
from pathlib import Path

sys.path.append(Path(__file__).parent.as_posix())
from Finetune_Config import Finetune_Config

sys.path.append((Path(__file__).parent.parent / 'Components').as_posix())
from Global_Config import (
    CHECKPOINT_BASE_PATH,
    LOG_BASE_PATH,
    Model_Name_Enum,
    Dataset_Name_Enum,
    Finetune_Type_Enum,
    Dataset_Eval_Metric_Enum,
    Dataset_Report_Metric_Enum,
    Dataset_Num_Label_Enum
)

sys.path.append((Path(__file__).parent.parent / 'Dataset_Tokenize').as_posix())
from Tokenize_Config import Model_Dataset_Length_Config_Enum



GRAD_ACCUM  = 1
SEED_LIST   = [1, 2, 3, 4, 5]
CONFIG_LIST = [
    ('Roberta_Base', 'Glue_Cola', 8, 1.00e-4, 15, 16, 0.1, 100), # model name, dataset name, lora rank, lr, epoch, batch size, regularization loss alpha, prune interval
    ('Roberta_Base', 'Glue_Cola', 8, 1.48e-4, 15, 16, 0.1, 100), # model name and dataset name should be a subset of Model_Name_Enum and Dataset_Name_Enum in Global_Config.py
    ('Roberta_Base', 'Glue_Cola', 8, 2.19e-4, 15, 16, 0.1, 100),
]



def init_finetune_config(model_name, dataset_name, lora_rank, seed, lr, epoch, batch_size, regularization_loss_alpha, prune_interval) -> Finetune_Config:
    config = Finetune_Config()

    additional_info = f'lr={lr:.2e}--epoch={epoch}--batch_size={batch_size}--regularization_loss_alpha={regularization_loss_alpha}--prune_interval={prune_interval}'
    config.task_tag = f'{model_name}--{dataset_name}--{seed}--{additional_info}'

    config.seed      = seed
    config.num_label = Dataset_Num_Label_Enum[dataset_name].value

    config.model_name      = Model_Name_Enum[model_name]
    config.dataset_name    = Dataset_Name_Enum[dataset_name]
    config.finetune_type   = Finetune_Type_Enum['Adaptive_Lora']
    config.checkpoint_path = CHECKPOINT_BASE_PATH / model_name / f'Lora_Rank={lora_rank}' / dataset_name / config.task_tag
    config.log_path        = LOG_BASE_PATH        / model_name / f'Lora_Rank={lora_rank}' / dataset_name / config.task_tag
    config.save_interval   = 20

    config.optimizer_type     = torch.optim.AdamW
    config.lr_scheduler_type  = 'linear'
    config.eval_metric        = Dataset_Eval_Metric_Enum[dataset_name].value
    config.report_metric_list = Dataset_Report_Metric_Enum[dataset_name].value

    if model_name.startswith('Roberta'):
        config.linear_name_to_apply_lora_list = ['query', 'key', 'value', 'attention.output', 'intermediate', 'output']
    elif model_name.startswith('Bart'):
        config.linear_name_to_apply_lora_list = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
    else:
        raise NotImplementedError(f'model name {model_name} not implemented')

    assert batch_size % GRAD_ACCUM   == 0, f'batch size should be divisible by grad accum value {GRAD_ACCUM}'
    config.lr                        = lr
    config.epoch                     = epoch
    config.batch_size                = batch_size // GRAD_ACCUM
    config.regularization_loss_alpha = regularization_loss_alpha
    config.grad_accumulation_step    = GRAD_ACCUM
    config.grad_clip                 = 1
    config.weight_decay              = 0.01
    config.warmup_ratio              = 0.06

    config.dataset_length = Model_Dataset_Length_Config_Enum[model_name].value[dataset_name].value

    config.lora_alpha      = 8
    config.lora_dropout    = 0.1
    config.fixed_lora_rank = 0

    config.adaptive_lora_start_rank             = math.ceil(lora_rank * 1.5)
    config.adaptive_lora_end_avg_rank           = lora_rank
    config.adaptive_lora_start_prune_step_ratio = 0.15
    config.adaptive_lora_end_prune_step_ratio   = 0.5
    config.adaptive_lora_prune_interval_step    = prune_interval
    config.adaptive_lora_sensitivity_beta       = 0.9
    config.adaptive_lora_eps                    = 1e-8

    config.generate_max_length = Model_Dataset_Length_Config_Enum[model_name].value[dataset_name].value[-1]
    config.generate_beam_size  = 3

    config.question_answering_max_answer_num      = 20
    config.question_answering_max_answer_length   = 30
    config.question_answering_no_answer_threshold = 0

    assert None not in config.to_dict().values(), f'all values in config should not be None, but got {config.to_dict()}'

    return config

def build_finetune_config_list() -> list[Finetune_Config]:
    finetune_config_list = []

    for model_name, dataset_name, lora_rank, lr, epoch, batch_size, regularization_loss_alpha, prune_interval in CONFIG_LIST:
        for seed in SEED_LIST:
            finetune_config_list.append(
                init_finetune_config(
                    model_name,
                    dataset_name,
                    lora_rank,
                    seed,
                    lr,
                    epoch,
                    batch_size,
                    regularization_loss_alpha,
                    prune_interval
                )
            )

    return finetune_config_list
