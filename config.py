import os

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
MODE_PATH = ROOT_PATH + '/bertmodel/'
DATA_PATH = ROOT_PATH + "/Jsondata/"

# BertTokenizerFast() 返回的dict的key
KEYTOKENID = "input_ids"
KEYTYPE = "token_type_ids"
KEYMASK = "attention_mask"
KEYOFFEST = "offset_mapping"
KEYTRIGLABEL = "triglabels"

# model document name
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
