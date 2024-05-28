from pathlib import Path
from aenum import Enum, NoAlias
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM
)



TOKENIZED_DATASET_BASE_PATH = Path(__file__).parent.parent.parent / 'Data'
CHECKPOINT_BASE_PATH        = Path(__file__).parent.parent.parent / 'Checkpoint'
LOG_BASE_PATH               = Path(__file__).parent.parent.parent / 'Log'



class Model_Name_Enum(Enum):
    '''Foundation type of all other enums about model, members of model enums should be a subset of this enum'''

    _settings_ = NoAlias

    Roberta_Base: str = 'Roberta_Base'
    Bart_Base: str    = 'Bart_Base'

class Model_Architecture_Enum(Enum):
    '''Foundation type of all other enums about model architecture, members of model architecture enums should be a subset of this enum'''

    _settings_ = NoAlias

    Encoder_Only: str    = 'Encoder_Only'
    Encoder_Decoder: str = 'Encoder_Decoder'
    Decoder_Only: str    = 'Decoder_Only'

class Dataset_Name_Enum(Enum):
    '''Foundation type of all other enums about dataset, members of dataset enums should be a subset of this enum'''

    _settings_ = NoAlias

    Glue_Cola: str            = 'Glue_Cola'
    Glue_Mnli_Matched: str    = 'Glue_Mnli_Matched'
    Glue_Mnli_Mismatched: str = 'Glue_Mnli_Mismatched'
    Glue_Mrpc: str            = 'Glue_Mrpc'
    Glue_Qnli: str            = 'Glue_Qnli'
    Glue_Qqp: str             = 'Glue_Qqp'
    Glue_Rte: str             = 'Glue_Rte'
    Glue_Sst2: str            = 'Glue_Sst2'
    Glue_Stsb: str            = 'Glue_Stsb'
    Squad_V1: str             = 'Squad_V1'
    Squad_V2: str             = 'Squad_V2'
    Xsum: str                 = 'Xsum'
    Cnn_Daily_Mail: str       = 'Cnn_Daily_Mail'

class Finetune_Type_Enum(Enum):
    '''Foundation type of all other enums about finetune type, members of finetune type enums should be a subset of this enum'''

    _settings_ = NoAlias

    Full_Finetune: str = 'Full_Finetune'
    Fixed_Lora: str    = 'Fixed_Lora'
    Adaptive_Lora: str = 'Adaptive_Lora'

class Problem_Type_Enum(Enum):
    '''Foundation type of all other enums about problem type, members of problem type enums should be a subset of this enum'''

    _settings_ = NoAlias

    Seq_Classification: str = 'Seq_Classification'
    Seq_Regression: str     = 'Seq_Regression'
    Question_Answering: str = 'Question_Answering'
    Summarization: str      = 'Summarization'



# Model configuration enums, if you want to add a new model, you should add a new member to all of these enums, and add name in Model_Name_Enum
class Model_Path_Enum(Enum):
    _settings_ = NoAlias

    Roberta_Base: str = 'roberta-base'
    Bart_Base: str    = 'facebook/bart-base'

class Model_Type_Enum(Enum):
    _settings_ = NoAlias

    Roberta_Base: str = Model_Architecture_Enum.Encoder_Only
    Bart_Base: str    = Model_Architecture_Enum.Encoder_Decoder

assert set(Model_Name_Enum.__members__) >= set(Model_Path_Enum.__members__), 'Model_Path_Enum should be a subset of Model_Name_Enum'
assert set(Model_Name_Enum.__members__) >= set(Model_Type_Enum.__members__), 'Model_Type_Enum should be a subset of Model_Name_Enum'



# Dataset configuration enums, if you want to add a new dataset, you should add a new member to all of these enums, and add name in Dataset_Name_Enum
class Dataset_Path_Enum(Enum):
    _settings_ = NoAlias

    Glue_Cola: tuple            = ('glue', 'cola')
    Glue_Mnli_Matched: tuple    = ('glue', 'mnli')
    Glue_Mnli_Mismatched: tuple = ('glue', 'mnli')
    Glue_Mrpc: tuple            = ('glue', 'mrpc')
    Glue_Qnli: tuple            = ('glue', 'qnli')
    Glue_Qqp: tuple             = ('glue', 'qqp')
    Glue_Rte: tuple             = ('glue', 'rte')
    Glue_Sst2: tuple            = ('glue', 'sst2')
    Glue_Stsb: tuple            = ('glue', 'stsb')
    Squad_V1: tuple             = ('squad', None)
    Squad_V2: tuple             = ('squad_v2', None)
    Xsum: tuple                 = ('EdinburghNLP/xsum', None)
    Cnn_Daily_Mail: tuple       = ('cnn_dailymail', '3.0.0')

class Dataset_Key_Enum(Enum):
    _settings_ = NoAlias

    Glue_Cola: tuple            = ('sentence',  None)
    Glue_Mnli_Matched: tuple    = ('premise',   'hypothesis')
    Glue_Mnli_Mismatched: tuple = ('premise',   'hypothesis')
    Glue_Mrpc: tuple            = ('sentence1', 'sentence2')
    Glue_Qnli: tuple            = ('question',  'sentence')
    Glue_Qqp: tuple             = ('question1', 'question2')
    Glue_Rte: tuple             = ('sentence1', 'sentence2')
    Glue_Sst2: tuple            = ('sentence',  None)
    Glue_Stsb: tuple            = ('sentence1', 'sentence2')
    Squad_V1: tuple             = ('question', 'context', 'answers', 'id')
    Squad_V2: tuple             = ('question', 'context', 'answers', 'id')
    Xsum: tuple                 = ('document', 'summary')
    Cnn_Daily_Mail: tuple       = ('article', 'highlights')

class Dataset_Problem_Type_Enum(Enum):
    _settings_ = NoAlias

    Glue_Cola: str            = Problem_Type_Enum.Seq_Classification
    Glue_Mnli_Matched: str    = Problem_Type_Enum.Seq_Classification
    Glue_Mnli_Mismatched: str = Problem_Type_Enum.Seq_Classification
    Glue_Mrpc: str            = Problem_Type_Enum.Seq_Classification
    Glue_Qnli: str            = Problem_Type_Enum.Seq_Classification
    Glue_Qqp: str             = Problem_Type_Enum.Seq_Classification
    Glue_Rte: str             = Problem_Type_Enum.Seq_Classification
    Glue_Sst2: str            = Problem_Type_Enum.Seq_Classification
    Glue_Stsb: str            = Problem_Type_Enum.Seq_Regression
    Squad_V1: str             = Problem_Type_Enum.Question_Answering
    Squad_V2: str             = Problem_Type_Enum.Question_Answering
    Xsum: str                 = Problem_Type_Enum.Summarization
    Cnn_Daily_Mail: str       = Problem_Type_Enum.Summarization

class Dataset_Eval_Metric_Enum(Enum):
    _settings_ = NoAlias

    Glue_Cola: str            = 'matthews_correlation'
    Glue_Mnli_Matched: str    = 'accuracy'
    Glue_Mnli_Mismatched: str = 'accuracy'
    Glue_Mrpc: str            = 'accuracy'
    Glue_Qnli: str            = 'accuracy'
    Glue_Qqp: str             = 'accuracy'
    Glue_Rte: str             = 'accuracy'
    Glue_Sst2: str            = 'accuracy'
    Glue_Stsb: str            = 'pearsonr'
    Squad_V1: str             = 'squad'
    Squad_V2: str             = 'squad_v2'
    Xsum: str                 = 'rouge'
    Cnn_Daily_Mail: str       = 'rouge'

class Dataset_Report_Metric_Enum(Enum):
    _settings_ = NoAlias

    Glue_Cola: list[str]            = ['matthews_correlation']
    Glue_Mnli_Matched: list[str]    = ['accuracy']
    Glue_Mnli_Mismatched: list[str] = ['accuracy']
    Glue_Mrpc: list[str]            = ['accuracy']
    Glue_Qnli: list[str]            = ['accuracy']
    Glue_Qqp: list[str]             = ['accuracy']
    Glue_Rte: list[str]             = ['accuracy']
    Glue_Sst2: list[str]            = ['accuracy']
    Glue_Stsb: list[str]            = ['pearsonr']
    Squad_V1: list[str]             = ['exact_match', 'f1']
    Squad_V2: list[str]             = ['exact', 'f1']
    Xsum: list[str]                 = ['rouge1', 'rouge2', 'rougeL']
    Cnn_Daily_Mail: list[str]       = ['rouge1', 'rouge2', 'rougeL']

class Dataset_Num_Label_Enum(Enum):
    _settings_ = NoAlias

    Glue_Cola: int            = 2
    Glue_Mnli_Matched: int    = 3
    Glue_Mnli_Mismatched: int = 3
    Glue_Mrpc: int            = 2
    Glue_Qnli: int            = 2
    Glue_Qqp: int             = 2
    Glue_Rte: int             = 2
    Glue_Sst2: int            = 2
    Glue_Stsb: int            = 1
    Squad_V1: int             = 2
    Squad_V2: int             = 2
    Xsum: int                 = 0
    Cnn_Daily_Mail: int       = 0

assert set(Dataset_Name_Enum.__members__) >= set(Dataset_Path_Enum.__members__),          'Dataset_Path_Enum should be a subset of Dataset_Name_Enum'
assert set(Dataset_Name_Enum.__members__) >= set(Dataset_Key_Enum.__members__),           'Dataset_Key_Enum should be a subset of Dataset_Name_Enum'
assert set(Dataset_Name_Enum.__members__) >= set(Dataset_Problem_Type_Enum.__members__),  'Dataset_Problem_Type_Enum should be a subset of Dataset_Name_Enum'
assert set(Dataset_Name_Enum.__members__) >= set(Dataset_Eval_Metric_Enum.__members__),   'Dataset_Eval_Metric_Enum should be a subset of Dataset_Name_Enum'
assert set(Dataset_Name_Enum.__members__) >= set(Dataset_Report_Metric_Enum.__members__), 'Dataset_Report_Metric_Enum should be a subset of Dataset_Name_Enum'
assert set(Dataset_Name_Enum.__members__) >= set(Dataset_Num_Label_Enum.__members__),     'Dataset_Num_Label_Enum should be a subset of Dataset_Name_Enum'



# different types of problems will fetch different models from Hugging Face.
class Problem_Type_Model_Type_Enum(Enum):
    _settings_ = NoAlias

    Seq_Classification: str = AutoModelForSequenceClassification
    Seq_Regression: str     = AutoModelForSequenceClassification
    Question_Answering: str = AutoModelForQuestionAnswering
    Summarization: str      = AutoModelForSeq2SeqLM

assert set(Problem_Type_Enum.__members__) >= set(Problem_Type_Model_Type_Enum.__members__), 'Problem_Type_Model_Type_Enum should be a subset of Problem_Type_Enum'
