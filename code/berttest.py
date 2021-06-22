from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__ + '/..')))

import json
from transformers import BertTokenizerFast
from pytorch_pretrained_bert.modeling import BertForTriggerClassification
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from config import ROOT_PATH, MODE_PATH, DATA_PATH, KEYOFFEST, KEYTOKENID, KEYTYPE, KEYMASK, KEYTRIGLABEL
import pprint
from collections import Counter
import torch
import logging
import time
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
candidate_queries = [
    ['what', 'is', 'the', 'trigger', 'in', 'the', 'event', '?'],  # 0 what is the trigger in the event?
    ['what', 'happened', 'in', 'the', 'event', '?'],  # 1 what happened in the event?
    ['trigger'],  # 2 trigger
    ['t'],  # 3 t
    ['action'],  # 4 action
    ['verb'],  # 5 verb
    ['null'],  # 6 null
]


class trigger_category_vocab(object):
    """docstring for trigger_category_vocab"""

    def __init__(self):
        self.category_to_index = dict()
        self.index_to_category = dict()
        self.trigerWords = list()
        self.counter = Counter()
        self.max_sent_length = 0

    def create_vocab(self, files_list, tokenizer: BertTokenizerFast):
        self.category_to_index["None"] = 0
        self.index_to_category[0] = "None"
        self.trigerWords.append(dict())
        for file in files_list:
            with open(file, "r", encoding='utf-8') as f:
                content = f.read()
                examples = json.loads(content)
                for example in examples:
                    events, sentence = example["golden-event-mentions"], example["sentence"]
                    lentok = len(tokenizer.tokenize(sentence))
                    if lentok > self.max_sent_length:
                        self.max_sent_length = lentok
                    for event in events:
                        event_type = event["event_type"]
                        self.counter[event_type] += 1
                        if event_type not in self.category_to_index:
                            index = len(self.index_to_category)
                            self.category_to_index[event_type] = index
                            self.index_to_category[index] = event_type
                            self.trigerWords.append(dict())
                        if "trigger" in event:
                            index = self.category_to_index[event_type]
                            word = event["trigger"]["text"]
                            if word in self.trigerWords[index]:
                                self.trigerWords[index][word] = self.trigerWords[index][word] + 1
                            else:
                                self.trigerWords[index][word] = 1

        # add [CLS] and query
        self.max_sent_length += 12


class AceEventObj(object):
    """A single set of features of data."""

    def __init__(self, sentence, events):
        # self.unique_id = unique_id
        # self.example_index = example_index
        # self.doc_span_index = doc_span_index
        self.sentence = sentence
        self.events = events


def getTokenIndex(offsetList: list, probindex, start, end):
    '''
    由原有index偏移获得二分查找获得对应编码后的token的index
    :param offsetList: return_offsets_mapping=True获得的"offset_mapping的list
    :param probindex:
    :param start:
    :param end:
    :return:
    '''
    if probindex >= offsetList[end][1]:
        return -1
    while start <= end:
        mid = (start + end) // 2
        if offsetList[mid][0] <= probindex and offsetList[mid][1] > probindex:
            tempres = mid
            return tempres
        elif offsetList[mid][1] <= probindex:
            start = mid + 1
        elif offsetList[mid][0] > probindex:
            end = mid - 1
    return end


def read_ACE_json(input_file):
    with open(input_file, "r", encoding='utf-8') as f:
        content = f.read()
        examples = json.loads(content)
        res = []
        for example in examples:
            sentence, events, = example["sentence"], example["golden-event-mentions"]
            if len(events) <= 0:
                continue
            res.append(AceEventObj(sentence=sentence, events=events))


def read_ace_examples(nth_query, input_file, tokenizer: BertTokenizerFast, maxLen=100):
    """
    :param nth_query:  use the nth kind query
    :param input_file:
    :param tokenizer: BertTokenizerFast
    :param category_vocab:
    :param is_training:
    :return:
    """
    """Read an ACE json file, transform to struct"""
    features = []
    examples = []
    sentence_id = 0
    with open(input_file, "r", encoding='utf-8') as f:
        content = f.read()
        examples = json.loads(content)
        reswithAns = []
        for index in range(len(examples)):
            sentence, events, = examples[index]["sentence"], examples[index]["golden-event-mentions"]
            if len(events) <= 0:
                continue
            tempdict = {}
            seq = " "
            query = seq.join(candidate_queries[nth_query])
            # sen_code = tokenizer(sentence, query, return_offsets_mapping=True, return_tensors="pt")
            # 要求补齐
            sen_code = tokenizer(sentence, query, return_offsets_mapping=True, max_length=maxLen,
                                 padding='max_length', truncation=True)
            tokenlen = len(sen_code["token_type_ids"])
            i = 0
            while i < tokenlen:
                if sen_code["token_type_ids"][i] == 1:
                    break
                i += 1
            sepindex = i - 1
            tempdict["orisen"] = sentence
            # tempdict["sentence"] =
            tempdict["sepindex"] = sepindex
            tempdict["token"] = sen_code
            tempdict["events"] = events

            i = 0
            # [tokenstart,tokenend]
            for event in tempdict["events"]:
                if 'arguments' in event:
                    j = 0
                    for arg in event["arguments"]:
                        probstart = arg["start"]
                        probend = arg["end"] - 1
                        tokenstart = getTokenIndex(sen_code[KEYOFFEST], probstart, 1, sepindex - 1)
                        tokenend = getTokenIndex(sen_code[KEYOFFEST], probend, 1, sepindex - 1)
                        tempdict["events"][i]["arguments"][j]["tokenstart"] = tokenstart
                        tempdict["events"][i]["arguments"][j]["tokenend"] = tokenend

                        # test
                        print("tokendecode:", tokenizer.decode(sen_code["input_ids"][tokenstart:tokenend + 1]),
                              "\norig:", arg["text"])
                        j += 1
                if "trigger" in event:
                    arg = event["trigger"]
                    probstart = arg["start"]
                    probend = arg["end"] - 1
                    tokenstart = getTokenIndex(sen_code[KEYOFFEST], probstart, 1, sepindex - 1)
                    tokenend = getTokenIndex(sen_code[KEYOFFEST], probend, 1, sepindex - 1)
                    tempdict["events"][i]["trigger"]["tokenstart"] = tokenstart
                    tempdict["events"][i]["trigger"]["tokenend"] = tokenend
                    # test
                    # print("tokendecode:", tokenizer.decode(sen_code["input_ids"][tokenstart:tokenend+1]),"\norig:", arg["text"])
                i += 1
            reswithAns.append(tempdict)
    return reswithAns


def read_ace_examples_Trig(nth_query, input_file, tokenizer: BertTokenizerFast, category_vocab: trigger_category_vocab):
    """
    :param nth_query:  use the nth kind query
    :param input_file:
    :param tokenizer: BertTokenizerFast
    :param category_vocab:
    :param is_training:
    :return:
    """
    """Read an ACE json file, transform to struct"""
    # print(category_vocab.category_to_index)
    reswithAns = []
    features = []
    with open(input_file, "r", encoding='utf-8') as f:
        content = f.read()
        examples = json.loads(content)
        sentence_id = 0
        for index in range(len(examples)):
            sentence, events, = examples[index]["sentence"], examples[index]["golden-event-mentions"]
            # if len(events) <= 0:
            #     continue
            tempdict = {}
            seq = " "
            query = seq.join(candidate_queries[nth_query])
            # sen_code = tokenizer(sentence, query, return_offsets_mapping=True, return_tensors="pt")
            # 要求补齐
            sen_code = tokenizer(sentence, query, return_offsets_mapping=True,
                                 max_length=category_vocab.max_sent_length,
                                 padding='max_length', truncation=True)
            tokenlen = len(sen_code["token_type_ids"])
            i = 0
            while i < tokenlen:
                if sen_code["token_type_ids"][i] == 1:
                    break
                i += 1
            sepindex = i - 1
            tempdict["orisen"] = sentence
            # tempdict["sentence"] =
            tempdict["sepindex"] = sepindex
            tempdict["token"] = sen_code
            tempdict["events"] = events
            assert len(sen_code[KEYTOKENID]) == category_vocab.max_sent_length
            assert len(sen_code[KEYTYPE]) == category_vocab.max_sent_length
            assert len(sen_code[KEYMASK]) == category_vocab.max_sent_length
            assert len(sen_code[KEYOFFEST]) == category_vocab.max_sent_length
            i = 0
            # [tokenstart,tokenend]
            labels = [0] * category_vocab.max_sent_length
            # rewrite tempdict[events] to add tokenstart and tokenend
            for event in tempdict["events"]:
                if 'arguments' in event:
                    j = 0
                    for arg in event["arguments"]:
                        probstart = arg["start"]
                        probend = arg["end"] - 1  # 闭区间
                        tokenstart = getTokenIndex(sen_code[KEYOFFEST], probstart, 1, sepindex - 1)
                        tokenend = getTokenIndex(sen_code[KEYOFFEST], probend, 1, sepindex - 1)
                        tempdict["events"][i]["arguments"][j]["tokenstart"] = tokenstart
                        tempdict["events"][i]["arguments"][j]["tokenend"] = tokenend

                        # test
                        # print("tokendecode:", tokenizer.decode(sen_code["input_ids"][tokenstart:tokenend + 1]),      "\norig:", arg["text"])
                        j += 1
                if "trigger" in event:
                    arg = event["trigger"]
                    probstart = arg["start"]
                    probend = arg["end"] - 1  # 闭区间
                    tokenstart = getTokenIndex(sen_code[KEYOFFEST], probstart, 1, sepindex - 1)
                    tokenend = getTokenIndex(sen_code[KEYOFFEST], probend, 1, sepindex - 1)
                    tempdict["events"][i]["trigger"]["tokenstart"] = tokenstart
                    tempdict["events"][i]["trigger"]["tokenend"] = tokenend
                    if 'event_type' in event:
                        caID = category_vocab.category_to_index[event["event_type"]]
                        for ac in range(tokenstart, tokenend + 1):
                            labels[ac] = caID
                        # labels[tokenstart] = caID
                        # if tokenend != tokenstart:
                        #     labels[tokenend] = caID + 10000
                    # test
                    # print("tokendecode:", tokenizer.decode(sen_code["input_ids"][tokenstart:tokenend + 1]), "\norig:",
                    #       arg["text"])
                i += 1
            tempdict["token"][KEYTRIGLABEL] = labels
            tempdict["id"] = sentence_id
            sentence_id += 1
            reswithAns.append(tempdict)
    return reswithAns


def loadReswithAns(reswithAns, batch_size=64):
    all_sentence_id = torch.tensor([f["id"] for f in reswithAns], dtype=torch.long)
    all_input_ids = torch.tensor([f["token"][KEYTOKENID] for f in reswithAns], dtype=torch.long)
    all_segmend_ids = torch.tensor([f["token"][KEYTYPE] for f in reswithAns], dtype=torch.long)
    # all_in_sentence = torch.tensor([f["token"][KEYTOKENID] for f in reswithAns], dtype=torch.long)
    all_input_mask = torch.tensor([f["token"][KEYMASK] for f in reswithAns], dtype=torch.long)
    all_labels = torch.tensor([f["token"][KEYTRIGLABEL] for f in reswithAns], dtype=torch.long)
    all_data = TensorDataset(all_sentence_id, all_input_ids, all_segmend_ids, all_input_mask,
                             all_labels)
    all_dataloader = DataLoader(all_data, batch_size=batch_size)
    return all_dataloader


pp = pprint.PrettyPrinter()
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained(MODE_PATH + "bert_base_uncased/", do_lower_case=True)
    category_vocab = trigger_category_vocab()
    filelist = [DATA_PATH+"ch\dev.json",DATA_PATH+r"ch\test.json",DATA_PATH+r"ch\train.json"]
    # category_vocab.create_vocab(filelist, tokenizer)
    category_vocab.create_vocab([DATA_PATH + "sample.json"], tokenizer)
    print(category_vocab.trigerWords)
    res = read_ace_examples_Trig(1, DATA_PATH + "sample.json", tokenizer, category_vocab)
    # res = sorted(res, key=lambda f: np.sum(f["token"][KEYMASK]), reverse=True)
    print(len(res[0]["token"][KEYTOKENID]))
    print(tokenizer.tokenize(res[0]["orisen"]))
    print(tokenizer.decode(res[0]["token"][KEYTOKENID]))
    print(res[0]["token"])
    # for val in res:
    #     print(val["token"][KEYTRIGLABEL])
    #     print(val["token"][KEYTYPE])
    #     # active_loss =torch.tensor( val["token"][KEYMASK]).view(-1) == 0
    #     # print(active_loss)
    #     # print(active_loss.size())
    #     # logits = torch.randn(active_loss.size()[0], 3)
    #     # active_logits = logits.view(-1, 3)[active_loss]
    #     # print(active_logits)
    #
    # traindata=loadReswithAns(res,batch_size=64)
    #
    # for _,val in enumerate(traindata):
    #     print(val)
    #     print("================")

    # print(len(res[0]["token"][KEYTOKENID]))
    # print(len(res[0]["token"][KEYOFFEST]))
    # print(len(res[0]["token"][KEYTRIGLABEL]))
    # print(res[0]["token"][KEYTRIGLABEL])
    # print(res[0]["token"][KEYTYPE])
    # print(res[0]["token"][KEYOFFEST])
    # print(res[0]["token"][KEYMASK])
    #
    # print(tokenizer.tokenize("New Questions About Attacking Iraq; Is Torturing Terrorists Necessary",
    #                          add_special_tokens=True, truncation=True))

    # model = BertForTriggerClassification.from_pretrained(MODE_PATH+"bert_base_uncased/",
    #                                                  num_labels=len(category_vocab.index_to_category))
    # model.to(device)
    # param_optimizer = list(model.named_parameters())
    # param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer
    #                 if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer
    #                 if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=4e-5,
    #                      warmup=0.1)
