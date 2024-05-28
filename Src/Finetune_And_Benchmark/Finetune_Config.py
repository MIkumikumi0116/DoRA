import sys
import dataclasses
from typing import Type
from pathlib import Path

sys.path.append((Path(__file__).parent.parent / 'Components').as_posix())
from Global_Config import (
    Model_Name_Enum,
    Dataset_Name_Enum,
    Finetune_Type_Enum
)



@dataclasses.dataclass
class Finetune_Config:
    '''
    everything for finetuning a model, hyperparameters, dataset, save path, etc, is included in this class
    default values are just placeholders, please change them to the actual values
    '''

    task_tag: str = 'model_name--dataset_name--random_seed--additional_info_if_needed or anything you like'

    seed: int      = None
    num_label: int = None

    model_name: Model_Name_Enum       = None
    dataset_name: Dataset_Name_Enum   = None
    finetune_type: Finetune_Type_Enum = None
    checkpoint_path: Path             = None
    log_path: Path                    = None
    save_interval: int                = None

    optimizer_type: Type                      = None
    lr_scheduler_type: str                    = None
    eval_metric: str                          = None
    report_metric_list: list[str]             = None
    linear_name_to_apply_lora_list: list[str] = None

    lr: float                        = None
    epoch: int                       = None
    batch_size: int                  = None
    regularization_loss_alpha: float = None
    grad_accumulation_step: int      = None
    grad_clip: float                 = None
    weight_decay: float              = None
    warmup_ratio: float              = None
    dataset_length: int              = None

    lora_alpha: int                             = None
    lora_dropout: float                         = None
    fixed_lora_rank: int                        = None
    adaptive_lora_start_rank: int               = None
    adaptive_lora_end_avg_rank: int             = None
    adaptive_lora_start_prune_step_ratio: float = None
    adaptive_lora_end_prune_step_ratio: float   = None
    adaptive_lora_prune_interval_step: int      = None
    adaptive_lora_sensitivity_beta: float       = None
    adaptive_lora_eps: float                    = None

    generate_max_length: int = None
    generate_beam_size: int  = None

    question_answering_max_answer_num: int        = None
    question_answering_max_answer_length: int     = None
    question_answering_no_answer_threshold: float = None

    def to_dict(self):
        self_dict = {}
        for field in dataclasses.fields(self):
            field_name  = field.name
            field_value = getattr(self, field_name)

            self_dict[field_name] = str(field_value)

        return self_dict
