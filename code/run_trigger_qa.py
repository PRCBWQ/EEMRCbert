"""Run QA model to detect triggers."""

from __future__ import absolute_import, division, print_function
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__ + '/..')))

import json
from transformers import BertTokenizerFast
from config import ROOT_PATH, MODE_PATH, DATA_PATH, KEYOFFEST, KEYTOKENID, KEYTYPE, KEYMASK, KEYTRIGLABEL, CONFIG_NAME, \
    WEIGHTS_NAME
from pytorch_pretrained_bert.modeling import BertForTriggerClassification, BertLSTMForTriggerClassification
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
import argparse
import collections
from collections import Counter
import copy
import logging
import math
import random
import time
import re
import string
from io import open
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

'''
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForTriggerClassification, BertLSTMForTriggerClassification
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)
'''
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
        self.counter = Counter()
        self.max_sent_length = 0

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
                        self.counter[event_type] += 1
                        if event_type not in self.category_to_index:
                            index = len(self.index_to_category)
                            self.category_to_index[event_type] = index
                            self.index_to_category[index] = event_type

        # add [CLS] and query
        self.max_sent_length += 12


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 # unique_id,
                 # example_index,
                 # doc_span_index,
                 sentence_id,
                 tokens,
                 # token_to_orig_map,
                 # token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 in_sentence,
                 labels):
        # self.unique_id = unique_id
        # self.example_index = example_index
        # self.doc_span_index = doc_span_index
        self.sentence_id = sentence_id
        self.tokens = tokens
        # self.token_to_orig_map = token_to_orig_map
        # self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.in_sentence = in_sentence
        self.labels = labels


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


def evaluate(args, eval_examples, category_vocab, model, device, eval_dataloader, pred_only=False):
    # eval_examples, , na_prob_thresh=1.0, pred_only=False):
    all_results = []
    model.eval()

    # get predictions
    pred_triggers = dict()
    for idx, (sentence_id, input_ids, segmend_ids, input_mask, labels) in enumerate(eval_dataloader):
        if pred_only and idx % 10 == 0:
            logger.info("Running test: %d / %d" % (idx, len(eval_dataloader)))
        sentence_id = sentence_id.tolist()
        input_ids = input_ids.to(device)
        segmend_ids = segmend_ids.to(device)
        input_mask = input_mask.to(device)
        labels = labels
        with torch.no_grad():
            # to get logits  evaluate do not need lables
            logits = model(input_ids, token_type_ids=segmend_ids, attention_mask=input_mask)

        for i, in_sent in enumerate(segmend_ids):
            logits_i = logits[i].detach().cpu()
            _, tag_seq = torch.max(logits_i, 1)  # 获得最大可能事件类型
            tag_seq = tag_seq.tolist()

            decoded_tag_seg = []
            for idj, j in enumerate(in_sent):  # 第一句为sentence token_type_id（segmend_ids）为0
                if j == 0:
                    decoded_tag_seg.append(category_vocab.index_to_category[tag_seq[idj]])
                else:
                    break
            sentence_triggers = []
            for offset, tag in enumerate(decoded_tag_seg):
                if tag != "None":
                    sentence_triggers.append([offset, tag])
            pred_triggers[sentence_id[i]] = sentence_triggers

    # get results (classification+ID)
    gold_triggers = dict()
    for eval_example in eval_examples:
        events = eval_example["events"]
        gold_sentence_triggers = []
        for event in events:
            gold_sentence_triggers.append([event["trigger"]["tokenstart"], event["event_type"]])
        gold_triggers[eval_example['id']] = gold_sentence_triggers

    gold_trigger_n, pred_trigger_n, true_positive_n = 0, 0, 0
    true_identi_n = 0
    for sentence_id in pred_triggers:
        gold_sentence_triggers = gold_triggers[sentence_id]
        pred_sentence_triggers = pred_triggers[sentence_id]
        # for pred_trigger_n
        for trigger in pred_sentence_triggers: pred_trigger_n += 1
        # for gold_trigger_n     
        for trigger in gold_sentence_triggers: gold_trigger_n += 1
        # for true_positive_n=
        for trigger in pred_sentence_triggers:
            if trigger in gold_sentence_triggers:
                true_positive_n += 1
                true_identi_n += 1
            else:
                tempoff = trigger[0]
                for ans in gold_sentence_triggers:
                    if ans[0] == tempoff:
                        true_identi_n += 1
                        break

    prec_c, recall_c, f1_c = 0, 0, 0
    if pred_trigger_n != 0:
        prec_c = 100.0 * true_positive_n / pred_trigger_n
    else:
        prec_c = 0
    if gold_trigger_n != 0:
        recall_c = 100.0 * true_positive_n / gold_trigger_n
    else:
        recall_c = 0
    if prec_c or recall_c:
        f1_c = 2 * prec_c * recall_c / (prec_c + recall_c)
    else:
        f1_c = 0

    # get results (identification)
    result = collections.OrderedDict()
    prec_i, recall_i, f1_i = 0, 0, 0
    if pred_trigger_n != 0:
        prec_i = 100.0 * true_identi_n / pred_trigger_n
    else:
        prec_i = 0
    if gold_trigger_n != 0:
        recall_i = 100.0 * true_identi_n / gold_trigger_n
    else:
        recall_i = 0
    if prec_i or recall_i:
        f1_i = 2 * prec_i * recall_i / (prec_i + recall_i)
    else:
        f1_i = 0
    result = collections.OrderedDict(
        [('prec_c', prec_c), ('recall_c', recall_c), ('f1_c', f1_c), ('prec_i', prec_i), ('recall_i', recall_i),
         ('f1_i', f1_i)])
    return result, pred_triggers


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    args.train_batch_size = \
        args.train_batch_size // args.gradient_accumulation_steps

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if args.do_train:
        assert (args.train_file is not None) and (args.dev_file is not None)

    if args.eval_test:
        assert args.test_file is not None
    else:
        assert args.dev_file is not None

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(args)

    tokenizer = BertTokenizerFast.from_pretrained(args.model, do_lower_case=args.do_lower_case)
    category_vocab = trigger_category_vocab()
    category_vocab.create_vocab([args.train_file, args.dev_file, args.test_file], tokenizer)

    if args.do_train or (not args.eval_test):
        eval_features = read_ace_examples_Trig(nth_query=args.nth_query, input_file=args.dev_file, tokenizer=tokenizer,
                                               category_vocab=category_vocab)
        if args.add_lstm:
            eval_features = sorted(eval_features, key=lambda f: np.sum(f["token"][KEYMASK]), reverse=True)
        logger.info("***** Dev *****")
        logger.info("  Num orig examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_dataloader = loadReswithAns(eval_features, args.eval_batch_size)

    if args.do_train:
        train_features = read_ace_examples_Trig(nth_query=args.nth_query, input_file=args.train_file,
                                                tokenizer=tokenizer,
                                                category_vocab=category_vocab)
        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f["token"][KEYMASK]))
        else:
            random.shuffle(train_features)
        if args.add_lstm:
            train_features = sorted(train_features, key=lambda f: np.sum(f["token"][KEYMASK]), reverse=True)
        train_dataloader = loadReswithAns(train_features, args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        logger.info("***** Train *****")
        logger.info("  Num orig examples = %d", len(train_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        eval_step = max(1, len(train_batches) // args.eval_per_epoch)
        best_result = None
        lrs = [args.learning_rate] if args.learning_rate else [1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
        for lr in lrs:
            if args.add_lstm:
                model = BertLSTMForTriggerClassification.from_pretrained(args.model,
                                                                         # cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
                                                                         num_labels=len(
                                                                             category_vocab.index_to_category))
            else:
                model = BertForTriggerClassification.from_pretrained(args.model,
                                                                     # cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
                                                                     num_labels=len(category_vocab.index_to_category))
            if args.fp16:
                model.half()
            model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)

            if not args.add_lstm:
                param_optimizer = list(model.named_parameters())
                param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
                no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in param_optimizer
                                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                    {'params': [p for n, p in param_optimizer
                                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
                if args.fp16:
                    try:
                        from apex.optimizers import FP16_Optimizer
                        from apex.optimizers import FusedAdam
                    except ImportError:
                        raise ImportError("Please install apex from https://www.github.com/nvidia/apex"
                                          "to use distributed and fp16 training.")
                    optimizer = FusedAdam(optimizer_grouped_parameters,
                                          lr=lr,
                                          bias_correction=False,
                                          max_grad_norm=1.0)
                    if args.loss_scale == 0:
                        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
                    else:
                        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
                else:
                    optimizer = BertAdam(optimizer_grouped_parameters,
                                         lr=lr,
                                         warmup=args.warmup_proportion,
                                         t_total=num_train_optimization_steps)
            else:
                optimizer = optim.SGD(model.parameters(), lr=args.lstm_lr, momentum=0.9, weight_decay=1e-6)

            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0
            global_step = 0
            start_time = time.time()
            for epoch in range(int(args.num_train_epochs)):
                model.train()
                logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
                if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                    random.shuffle(train_batches)
                for step, batch in enumerate(train_batches):
                    if n_gpu == 1:
                        batch = tuple(t.to(device) for t in batch)
                    index, input_ids, segment_ids, input_mask, labels = batch
                    loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=labels)
                    if n_gpu > 1:
                        loss = loss.mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            lr_this_step = lr * \
                                           warmup_linear(global_step / num_train_optimization_steps,
                                                         args.warmup_proportion)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                    if (step + 1) % eval_step == 0 or step == 0:
                        # logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                        #     epoch, step + 1, len(train_batches), time.time() - start_time, tr_loss / nb_tr_steps))

                        save_model = False
                        if args.do_eval:
                            result, _ = evaluate(args, eval_features, category_vocab, model, device, eval_dataloader)
                            model.train()
                            result['global_step'] = global_step
                            result['epoch'] = epoch
                            result['learning_rate'] = lr
                            result['batch_size'] = args.train_batch_size
                            if args.add_lstm:
                                logger.info("        p: %.2f, r: %.2f, f1: %.2f" % (
                                    result["prec"], result["recall"], result["f1"]))
                            if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                                best_result = result
                                save_model = True
                                logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                                    epoch, step + 1, len(train_batches), time.time() - start_time,
                                           tr_loss / nb_tr_steps))
                                logger.info(
                                    "!!! Best dev %s (lr=%s, epoch=%d): p_c: %.2f, r_c: %.2f, f1_c: %.2f, p_i: %.2f, r_i: %.2f, f1_i: %.2f" %
                                    (args.eval_metric, str(lr), epoch, result["prec_c"], result["recall_c"],
                                     result["f1_c"], result["prec_i"], result["recall_i"], result["f1_i"]))
                        else:
                            save_model = True
                        if (int(args.num_train_epochs) - epoch < 3 and (step + 1) / len(
                                train_batches) > 0.7) or step == 0:
                            save_model = True
                        else:
                            save_model = False
                        if save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            subdir = os.path.join(args.output_dir,
                                                  "epoch{epoch}-step{step}".format(epoch=epoch, step=step))
                            if not os.path.exists(subdir):
                                os.makedirs(subdir)
                            output_model_file = os.path.join(subdir, WEIGHTS_NAME)
                            output_config_file = os.path.join(subdir, CONFIG_NAME)
                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                            tokenizer.save_vocabulary(subdir)
                            if best_result:
                                with open(os.path.join(args.output_dir, "eval_results.txt"), "w") as writer:
                                    # for key in sorted(best_result.keys()):
                                    for key in best_result:
                                        writer.write("%s = %s\n" % (key, str(best_result[key])))

            del model

    if args.do_eval:
        if args.eval_test:
            eval_features = read_ace_examples_Trig(nth_query=args.nth_query, input_file=args.test_file,
                                                   tokenizer=tokenizer, category_vocab=category_vocab)
            if args.add_lstm:
                eval_features = sorted(eval_features, key=lambda f: np.sum(f.input_mask), reverse=True)
            logger.info("***** Test *****")
            logger.info("  Num orig examples = %d", len(eval_features))
            logger.info("  Batch size = %d", args.eval_batch_size)
            eval_dataloader = loadReswithAns(eval_features, args.eval_batch_size)

        # BertForTriggerClassification.from_pretrained(args.model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE, num_labels=len(category_vocab.index_to_category))
        if args.add_lstm:
            model = BertLSTMForTriggerClassification.from_pretrained(args.model_dir,
                                                                     num_labels=len(category_vocab.index_to_category))
        else:
            model = BertForTriggerClassification.from_pretrained(args.model_dir,
                                                                 num_labels=len(category_vocab.index_to_category))
        if args.fp16:
            model.half()
        model.to(device)
        result, preds = evaluate(args, eval_features, category_vocab, model, device, eval_dataloader)

        with open(os.path.join(args.model_dir, "test_results.txt"), "w") as writer:
            for key in result:
                writer.write("%s = %s\n" % (key, str(result[key])))
        with open(os.path.join(args.model_dir, "trigger_predictions.json"), "w") as writer:
            for line in preds:
                writer.write(json.dumps(preds[line], default=int) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--model_dir", default="trigger_qa_output/epoch0-step0", type=str, required=True,
                        help="eval/test model")
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--dev_file", default=None, type=str)
    parser.add_argument("--test_file", default=None, type=str)
    parser.add_argument("--eval_per_epoch", default=10, type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_test", action='store_true', help='Wehther to run eval on the test set.')
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=None, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--eval_metric", default='f1_c', type=str)
    parser.add_argument("--train_mode", type=str, default='random_sorted',
                        choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--add_lstm", action='store_true', help="Whether to add LSTM on top of BERT.")
    parser.add_argument("--lstm_lr", default=None, type=float, help="The initial learning rate for lstm Adam.")
    parser.add_argument("--nth_query", default=0, type=int, help="use n-th candidate query")
    args = parser.parse_args()

    main(args)
