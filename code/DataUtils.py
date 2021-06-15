import json
from transformers import BertTokenizerFast
from config import ROOT_PATH, MODE_PATH, DATA_PATH, KEYOFFEST, KEYTOKENID, KEYTYPE, KEYMASK, KEYTRIGLABEL, CONFIG_NAME, \
    WEIGHTS_NAME,candidate_queries
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
class Event_category_vocab(object):
    """docstring for trigger_category_vocab"""

    def __init__(self):
        self.category_to_index = dict()
        self.index_to_category = dict()
        self.Eventcounter = Counter()
        self.Rolecounter = Counter()
        self.max_sent_length = 0
        self.eventRoledic = dict()
        self.totalRoledic = dict()
        self.trigerWords = list()
    def create_vocab(self, files_list, tokenizer: BertTokenizerFast):
        self.category_to_index["None"] = 0
        self.index_to_category[0] = "None"
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
                        self.Eventcounter[event_type] += 1
                        if event_type not in self.category_to_index:
                            index = len(self.index_to_category)
                            self.category_to_index[event_type] = index
                            self.index_to_category[index] = event_type
                            self.eventRoledic[event_type] = dict()
                            self.trigerWords.append(dict())
                        if "trigger" in event:
                            index = self.category_to_index[event_type]
                            word = event["trigger"]["text"]
                            if word in self.trigerWords[index]:
                                self.trigerWords[index][word] = self.trigerWords[index][word] + 1
                            else:
                                self.trigerWords[index][word] = 1
                        if "arguments" not in event:
                            continue
                        arguments = event["arguments"]
                        for attri in arguments:
                            role = attri["role"]
                            if role not in self.eventRoledic[event_type]:
                                self.eventRoledic[event_type][role] = 1
                            else:
                                self.eventRoledic[event_type][role] += 1
                            if role not in self.totalRoledic:
                                self.totalRoledic[role] = 1
                            else:
                                self.totalRoledic[role] += 1
        # add [CLS] and query
        self.max_sent_length += 12
def getTokenIndex(offsetList: list, probindex, start, end):
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
def getsepandlen(token_type_ids:list):
    tokenlen = len(token_type_ids)
    i = 0
    while i < tokenlen:
        if token_type_ids[i] == 1:
            break
        i += 1
    sepindex = i - 1
    while i < tokenlen:
        if token_type_ids[i] == 0:
            break
        i += 1
    setlen = i-1
    return sepindex, setlen

def read_ACE_toTrigExamples(nth_query,input_file, tokenizer: BertTokenizerFast, category_vocab: Event_category_vocab):
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
            sen_code = tokenizer(query, sentence, return_offsets_mapping=True,
                                 max_length=category_vocab.max_sent_length,
                                 padding='max_length', truncation=True)
            sepindex, seglen = getsepandlen(sen_code[KEYTYPE])
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
                        tokenstart = getTokenIndex(sen_code[KEYOFFEST], probstart, sepindex+1, seglen)
                        tokenend = getTokenIndex(sen_code[KEYOFFEST], probend, sepindex+1, seglen)
                        tempdict["events"][i]["arguments"][j]["tokenstart"] = tokenstart
                        tempdict["events"][i]["arguments"][j]["tokenend"] = tokenend

                        # test
                        # print("tokendecode:", tokenizer.decode(sen_code["input_ids"][tokenstart:tokenend + 1]),      "\norig:", arg["text"])
                        j += 1
                if "trigger" in event:
                    arg = event["trigger"]
                    probstart = arg["start"]
                    probend = arg["end"] - 1  # 闭区间
                    tokenstart = getTokenIndex(sen_code[KEYOFFEST], probstart, sepindex+1, seglen)
                    tokenend = getTokenIndex(sen_code[KEYOFFEST], probend, sepindex+1, seglen)
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
def loadReswithAns(reswithAns, batch_size=32):
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
