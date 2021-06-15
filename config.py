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

# Triger query
candidate_queries = [
    ['what', 'is', 'the', 'trigger', 'in', 'the', 'event', '?'],  # 0 what is the trigger in the event?
    ['what', 'happened', 'in', 'the', 'event', '?'],  # 1 what happened in the event?
    ['trigger'],  # 2 trigger
    ['t'],  # 3 t
    ['action'],  # 4 action
    ['verb'],  # 5 verb
    ['null'],  # 6 null
]
