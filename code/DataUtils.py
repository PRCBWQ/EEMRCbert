import json
import os

from transformers import BertTokenizerFast
from config import ROOT_PATH, MODE_PATH, DATA_PATH, KEYOFFEST, KEYTOKENID, KEYTYPE, KEYMASK, KEYTRIGLABEL, CONFIG_NAME, \
    WEIGHTS_NAME, candidate_queries
from collections import Counter
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


class Event_category_vocab(object):
    """docstring for trigger_category_vocab"""

    def __init__(self):
        self.category_to_index = dict()
        self.index_to_category = dict()
        self.max_sent_length = 0

        self.eventRoledic = dict()  # the role in a kind of event
        self.totalRoledic = dict()  # the count of each role
        self.Eventcounter = Counter()  # the count of each category
        self.triggerWords = dict()  # the triggerWords of each kind event has

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
                        self.Eventcounter["total"] += 1
                        self.Eventcounter[event_type] += 1
                        if event_type not in self.category_to_index:
                            index = len(self.index_to_category)
                            self.category_to_index[event_type] = index
                            self.index_to_category[index] = event_type
                            self.eventRoledic[event_type] = dict()
                            self.triggerWords[index] = dict()
                        if "trigger" in event:
                            index = self.category_to_index[event_type]
                            word = event["trigger"]["text"]
                            if word in self.triggerWords[index]:
                                self.triggerWords[index][word] = self.triggerWords[index][word] + 1
                            else:
                                self.triggerWords[index][word] = 1
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

    def save_result(self, output="Statistic/"):
        if not os.path.isdir(DATA_PATH + output):
            os.mkdir(DATA_PATH + output)
        with open(DATA_PATH + output + "_category.json", "w", encoding='utf-8') as f:
            json.dump(self.category_to_index, f, indent=4, ensure_ascii=False)  # File use dump and Main Memory us dumps
        with open(DATA_PATH + output + "_trigger.json", "w", encoding='utf-8') as f:
            json.dump(self.triggerWords, f, indent=4, ensure_ascii=False)
        with open(DATA_PATH + output + "_Role.json", "w", encoding='utf-8') as f:
            json.dump(self.eventRoledic, f, indent=4, ensure_ascii=False)
        with open(DATA_PATH + output + "_totalRole.json", "w", encoding='utf-8') as f:
            json.dump(self.totalRoledic, f, indent=4, ensure_ascii=False)
        with open(DATA_PATH + output + "_totalEvent.json", "w", encoding='utf-8') as f:
            json.dump(self.Eventcounter, f, indent=4, ensure_ascii=False)


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


def getsepandlen(token_type_ids: list):
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
    setlen = i - 1
    return sepindex, setlen


def read_ACE_toTrigExamples(nth_query, input_file, tokenizer: BertTokenizerFast, category_vocab: Event_category_vocab):
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
            sen_code = tokenizer(query, sentence, return_offsets_mapping=True, add_special_tokens=True,
                                 max_length=category_vocab.max_sent_length,
                                 padding='max_length', truncation=True)
            sen_code['senlist'] = tokenizer.tokenize(query,sentence, add_special_tokens=True)
            # print(sen_code['senlist'])
            sepindex, seglen = getsepandlen(sen_code[KEYTYPE]) #[sep] index
            # print("sepindex:"+str(sepindex))
            # print("seglen:"+str(seglen))
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
                        tokenstart = getTokenIndex(sen_code[KEYOFFEST], probstart, sepindex + 1, seglen-1)
                        tokenend = getTokenIndex(sen_code[KEYOFFEST], probend, sepindex + 1, seglen-1)
                        tempdict["events"][i]["arguments"][j]["tokenstart"] = tokenstart
                        tempdict["events"][i]["arguments"][j]["tokenend"] = tokenend

                        # test
                        # print("tokendecode:", tokenizer.decode(sen_code["input_ids"][tokenstart:tokenend + 1]),      "\norig:", arg["text"])
                        j += 1
                if "trigger" in event:
                    arg = event["trigger"]
                    probstart = arg["start"]
                    probend = arg["end"] - 1  # 闭区间
                    tokenstart = getTokenIndex(sen_code[KEYOFFEST], probstart, sepindex + 1, seglen-1)
                    tokenend = getTokenIndex(sen_code[KEYOFFEST], probend, sepindex + 1, seglen-1)
                    tempdict["events"][i]["trigger"]["tokenstart"] = tokenstart
                    tempdict["events"][i]["trigger"]["tokenend"] = tokenend
                    # print("tokenstart:"+str(tokenstart))
                    # print("tokenend:" + str(tokenend))
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


if __name__ == "__main__":
    filelist = [DATA_PATH + "ch\dev.json", DATA_PATH + r"ch\test.json", DATA_PATH + r"ch\train.json"]
    tokenizer = BertTokenizerFast.from_pretrained(MODE_PATH + "bert_base_uncased/", do_lower_case=True)
    category_vocab = Event_category_vocab()
    category_vocab.create_vocab(filelist, tokenizer)
    # category_vocab.save_result()
    res = read_ACE_toTrigExamples(1, DATA_PATH + "sample.json", tokenizer, category_vocab)
    print("#################################")
    for temp in res:
        print(temp['token'][KEYOFFEST])
        print(temp['token'][KEYTOKENID])
        print(temp['token'][KEYTYPE])
        print(temp['token'][KEYMASK])
        print(temp['token'][KEYTRIGLABEL])
