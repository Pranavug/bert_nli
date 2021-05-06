import sys
sys.path.append('./apex')

"""
The system trains BERT on the SNLI + MultiNLI (AllNLI) dataset with softmax loss function.
At every 1000 training steps, the model is evaluated on the dev set.
"""
import time
import logging
from datetime import datetime
import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np
from transformers import *
import math
import argparse
import random
import copy
import os
from nltk.tokenize import word_tokenize

from utils.nli_data_reader import NLIDataReader
from utils.logging_handler import LoggingHandler
from bert_nli import BertNLIModel, POOLING_CHOICES
from test_trained_model import evaluate
from losses import BlendedLoss, MAIN_LOSS_CHOICES, OnlineTripletLoss
from evaluate import evaluate_knn, evaluate_svm, evaluate_protoNN

# constants
DEVICE_CHOICES = ("cuda:0", "cuda:1", "cuda:2", "cuda:3")
acc_pruned_list = []
num_layers_pruned = 0
best_acc = -1


def get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
    """
    Returns the correct learning rate scheduler
    """
    scheduler = scheduler.lower()
    if scheduler=='constantlr':
        return get_constant_schedule(optimizer)
    elif scheduler=='warmupconstant':
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif scheduler=='warmuplinear':
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    elif scheduler=='warmupcosine':
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    elif scheduler=='warmupcosinewithhardrestarts':
        return get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
    else:
        raise ValueError("Unknown scheduler {}".format(scheduler))


def train(model, optimizer, scheduler, train_data, dev_data, batch_size, fp16, checkpoint, gpu, max_grad_norm, best_acc_1, loss_type, cross_entropy_flag, device):
    global acc_pruned_list
    global num_layers_pruned
    global best_acc

    if 'triplet' in loss_type:
        loss_fn = OnlineTripletLoss(loss_type, 1.0, device)
    else:
        loss_fn = BlendedLoss(loss_type, cross_entropy_flag, device)

    step_cnt = 0
    best_model_weights = None
    torch.cuda.empty_cache() # releases all unoccupied cached memory

    for pointer in tqdm(range(0, len(train_data), batch_size),desc='training', position=0, leave=True):
        model.train() # model was in eval mode in evaluate(); re-activate the train mode
        optimizer.zero_grad() # clear gradients first
        # torch.cuda.empty_cache() # releases all unoccupied cached memory

        step_cnt += 1
        sent_pairs = []
        labels = []
        for i in range(pointer, pointer+batch_size):
            if i >= len(train_data): break
            sents = train_data[i].get_texts()
            if len(word_tokenize(' '.join(sents))) > 300: continue
            sent_pairs.append(sents)
            labels.append(train_data[i].get_label())
        logits, reps = model.ff(sent_pairs,checkpoint)
        if logits is None: continue
        true_labels = torch.LongTensor(labels)
        # print("True labels", true_labels)
        if gpu:
            true_labels = true_labels.to(device)

        try:
            blended_loss, losses = loss_fn.calculate_loss(true_labels, logits, logits)
        except RuntimeError as e:
            continue

        # back propagate
        if fp16:
            with amp.scale_loss(blended_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
        else:
            blended_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # update weights
        optimizer.step()

        # update training rate
        scheduler.step()

        if step_cnt%2000 == 0:
            acc = evaluate(model,dev_data,checkpoint,mute=True)
            logging.info('==> step {} dev acc: {}, best acc: {}, best acc list: {}, loss: {}'.format(step_cnt, acc, best_acc, acc_pruned_list, losses))
            if acc > best_acc:
                best_acc = acc
                best_model_weights = copy.deepcopy(model.cpu().state_dict())
                model.to(device)

    if model.num_layers < 1:
        acc_pruned_list.append(best_acc)
        model.num_layers -= 1

        best_acc = -1
        num_layers_pruned += 1

        print("Model no. of layers:", model.num_layers, "No. of layers pruned:", num_layers_pruned)

    return best_model_weights


def parse_args():
    ap = argparse.ArgumentParser("arguments for bert-nli training")
    ap.add_argument('-b','--batch_size',type=int,default=17,help='batch size')
    ap.add_argument('-ep','--epoch_num',type=int,default=1,help='epoch num')
    ap.add_argument('--fp16',type=int,default=0,help='use apex mixed precision training (1) or not (0); do not use this together with checkpoint')
    ap.add_argument('--check_point','-cp',type=int,default=0,help='use checkpoint (1) or not (0); this is required for training bert-large or larger models; do not use this together with apex fp16')
    ap.add_argument('--gpu',type=int,default=1,help='use gpu (1) or not (0)')
    ap.add_argument('-ss','--scheduler_setting',type=str,default='WarmupLinear',choices=['WarmupLinear','ConstantLR','WarmupConstant','WarmupCosine','WarmupCosineWithHardRestarts'])
    ap.add_argument('-tm','--trained_model',type=str,default='None',help='path to the trained model; make sure the trained model is consistent with the model you want to train')
    ap.add_argument('-mg','--max_grad_norm',type=float,default=1.,help='maximum gradient norm')
    ap.add_argument('-wp','--warmup_percent',type=float,default=0.1,help='how many percentage of steps are used for warmup')
    ap.add_argument('-bt','--bert_type',type=str,default='bert-base',help='transformer (bert) pre-trained model you want to use', choices=['bert-base','bert-large','albert-base-v2','albert-large-v2'])
    ap.add_argument('--hans',type=int,default=0,help='use hans data (1) or not (0)')
    ap.add_argument('-rl','--reinit_layers',type=int,default=0,help='reinitialise the last N layers')
    ap.add_argument('-fl','--freeze_layers',type=int,default=0,help='whether to freeze all but the lasat few layers (1) or not (0)')
    ap.add_argument('--cross_entropy_flag', type=bool)
    ap.add_argument('--loss_type', type=str, default='n-pair', choices=MAIN_LOSS_CHOICES)
    ap.add_argument('--pool_type', type=str, default='average', choices=POOLING_CHOICES)
    ap.add_argument('--device', type=str, default='cuda:0', choices=DEVICE_CHOICES)
    ap.add_argument('--num_layers',type=int,default=12,help='No. of encoder layers for BERT')
    ap.add_argument('--output_dir', type=str, default='temp')
    ap.add_argument('--bert_layers', nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    ap.add_argument('--proj_dim',type=int,default=60)
    ap.add_argument('--num_proj',type=int,default=400)

    args = ap.parse_args()
    return args.batch_size, args.epoch_num, args.fp16, args.check_point, args.gpu,  args.scheduler_setting, args.max_grad_norm, args.warmup_percent, args.bert_type, args.trained_model, args.hans, args.reinit_layers, args.freeze_layers, args.loss_type, args.cross_entropy_flag, args.pool_type, args.device, args.num_layers, args.output_dir, args.bert_layers, args.proj_dim, args.num_proj


if __name__ == '__main__':

    batch_size, epoch_num, fp16, checkpoint, gpu, scheduler_setting, max_grad_norm, warmup_percent, bert_type, trained_model, hans, reinit_layers, freeze_layers, loss_type, cross_entropy_flag, pool_type, device, num_layers, output_dir, bert_layers, proj_dim, num_proj = parse_args()
    fp16 = bool(fp16)
    gpu = bool(gpu)
    hans = bool(hans)
    checkpoint = bool(checkpoint)
    if trained_model=='None': trained_model=None

    print('=====Arguments=====')
    print('bert type:\t{}'.format(bert_type))
    print('trained model path:\t{}'.format(trained_model))
    print('batch size:\t{}'.format(batch_size))
    print('epoch num:\t{}'.format(epoch_num))
    print('fp16:\t{}'.format(fp16))
    print('check_point:\t{}'.format(checkpoint))
    print('gpu:\t{}'.format(gpu))
    print('scheduler setting:\t{}'.format(scheduler_setting))
    print('max grad norm:\t{}'.format(max_grad_norm))
    print('warmup percent:\t{}'.format(warmup_percent))
    print('using hans:\t{}'.format(hans))
    print('No. of layers:\t{}'.format(num_layers))
    print('Pooling type:\t{}'.format(pool_type))
    print('Loss type:\t{}'.format(loss_type))
    print('Device:\t{}'.format(device))
    print('Output Dir:\t{}'.format(output_dir))
    print('=====Arguments=====')

    label_num = 3
    if hans:
        model_save_path = 'output/nli_hans_{}-{}'.format(bert_type,datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        model_save_path = 'output/{}_{}-{}'.format(output_dir, bert_type,datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    print('model save path', model_save_path)

    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
    #### /print debug information to stdout

    # Read the dataset
    if hans:
        nli_reader = NLIDataReader('datasets/Hans')
        hans_data = nli_reader.get_hans_examples('heuristics_train_set.txt')
    else:
        hans_data = []

    nli_reader = NLIDataReader('datasets/AllNLI')
    train_num_labels = nli_reader.get_num_labels()
    msnli_data = nli_reader.get_examples('train.gz',max_examples=-1)

    all_data = msnli_data + hans_data
    random.shuffle(all_data)
    train_num = int(len(all_data)*0.95)
    train_data = all_data[:train_num]
    dev_data = all_data[train_num:]

    logging.info('train data size {}'.format(len(train_data)))
    logging.info('dev data size {}'.format(len(dev_data)))

    total_steps = math.ceil(epoch_num*len(train_data)*1./batch_size)
    warmup_steps = int(total_steps*warmup_percent)

    model = BertNLIModel(gpu=gpu,batch_size=batch_size,bert_type=bert_type,model_path=trained_model, reinit_num=reinit_layers, freeze_layers=freeze_layers, pool_type=pool_type, device=device, num_layers=num_layers, bert_layers=bert_layers)
    optimizer = AdamW(model.parameters(),lr=1e-5,eps=1e-6,correct_bias=False)
    scheduler = get_scheduler(optimizer, scheduler_setting, warmup_steps=warmup_steps, t_total=total_steps)
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    best_model_dic = None
    os.makedirs(model_save_path, exist_ok=True)

    train_start_ts = time.time()
    for ep in range(epoch_num):
        logging.info('\n=====epoch {}/{}====='.format(ep,epoch_num))
        model_dic = train(model, optimizer, scheduler, train_data, dev_data, batch_size, fp16, checkpoint, gpu, max_grad_norm, best_acc, loss_type, cross_entropy_flag, device)
        if model_dic is not None:
            best_model_dic = model_dic
            model.save(model_save_path,best_model_dic)

    train_end_ts = time.time()

    # assert best_model_dic is not None

    # for testing load the best model
    model.load_model(best_model_dic)
    logging.info('\n=====Training finished. Now start test=====')

    if hans:
        nli_reader = NLIDataReader('datasets/Hans')
        hans_test_data = nli_reader.get_hans_examples('heuristics_evaluation_set.txt')
    else:
        hans_test_data = []

    nli_reader = NLIDataReader('datasets/AllNLI')
    msnli_test_data = nli_reader.get_examples('dev.gz', max_examples=-1)
    train_data = nli_reader.get_examples('train.gz',max_examples=-1) # 50000
    random.shuffle(train_data)

    test_data = msnli_test_data + hans_test_data
    # test_data = dev_data

    evaluate_protoNN(model, train_data, test_data, device, batch_size, checkpoint, proj_dim, num_proj)

    logging.info('test data size: {}'.format(len(test_data)))
    predict_start_ts = time.time()
    test_acc = evaluate(model, test_data, checkpoint)
    predict_end_ts = time.time()
    logging.info('accuracy on test set: {}'.format(test_acc))
    print("Training time:", train_end_ts-train_start_ts, "Prediction time:", predict_end_ts-predict_start_ts)

    evaluate_knn(model, train_data, test_data, device, batch_size, checkpoint, n_neighbors=10)

    train_data = nli_reader.get_examples('train.gz',max_examples=100000) # 25000
    random.shuffle(train_data)
    evaluate_svm(model, train_data, test_data, device, batch_size, checkpoint)

    if model_save_path is not None:
        os.makedirs(model_save_path, exist_ok=True)
        if os.listdir(model_save_path):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(
                model_save_path))
