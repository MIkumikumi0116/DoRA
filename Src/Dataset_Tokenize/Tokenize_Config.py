import sys
from pathlib import Path
from aenum import Enum, NoAlias, EnumType

sys.path.append((Path(__file__).parent.parent / 'Components').as_posix())
from Global_Config import Model_Name_Enum, Dataset_Name_Enum



class Roberta_Base_Dataset_Length_Enum(Enum):
    _settings_ = NoAlias

    Glue_Cola: tuple[int]            = (128,)
    Glue_Mnli_Matched: tuple[int]    = (128,)
    Glue_Mnli_Mismatched: tuple[int] = (128,)
    Glue_Mrpc: tuple[int]            = (128,)
    Glue_Qnli: tuple[int]            = (128,)
    Glue_Qqp: tuple[int]             = (128,)
    Glue_Rte: tuple[int]             = (128,)
    Glue_Sst2: tuple[int]            = (128,)
    Glue_Stsb: tuple[int]            = (128,)
    Squad_V1: tuple[int]             = (384, 128) # (seq_length, stride_length)
    Squad_V2: tuple[int]             = (384, 128)

class Bart_Base_Data_Length_Enum(Enum):
    _settings_ = NoAlias

    Xsum: tuple[int]           = (768,  64) # (text_length, summary_length)
    Cnn_Daily_Mail: tuple[int] = (1024, 160)

assert set(Dataset_Name_Enum.__members__) >= set(Roberta_Base_Dataset_Length_Enum.__members__), 'Roberta_Base_Dataset_Length_Enum should be a subset of Dataset_Name_Enum'
assert set(Dataset_Name_Enum.__members__) >= set(Bart_Base_Data_Length_Enum.__members__),       'Bart_Base_Data_Length_Enum should be a subset of Dataset_Name_Enum'



class Model_Dataset_Length_Config_Enum(Enum):
    _settings_ = NoAlias

    Roberta_Base: EnumType = Roberta_Base_Dataset_Length_Enum
    Bart_Base: EnumType    = Bart_Base_Data_Length_Enum

assert set(Model_Name_Enum.__members__) >= set(Model_Dataset_Length_Config_Enum.__members__), 'Model_Dataset_Length_Config_Enum should be a subset of Model_Name_Enum'
