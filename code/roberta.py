import os
import re
import time
import pickle
import numpy as np
import pandas as pd
from functools import reduce

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn

from transformers import AdamW, BertTokenizer
from transformers import BertForSequenceClassification


class PassionDataset(object):
    def __init__(self, data_list, tokenizer):
        self.data_list = data_list
        self.tokenizer = tokenizer

    def __len__(self,):
        return len(self.data_list)

    def __getitem__(self, index):
        audio_path, asr_text, label = self.data_list[index]
        audio_name = os.path.basename(audio_path)
        text_input = self.tokenizer(asr_text)
        return {'text_input':text_input,'label':label,'audio_name':audio_name}


def metric_helper(pred_y, true_y):
    pos_pre = precision_score(y_true=true_y, y_pred=pred_y, pos_label=1)
    pos_rec = recall_score(y_true=true_y, y_pred=pred_y, pos_label=1)
    pos_f1 = f1_score(y_true=true_y, y_pred=pred_y, pos_label=1)
    neg_pre = precision_score(y_true=true_y, y_pred=pred_y, pos_label=0)
    neg_rec = recall_score(y_true=true_y, y_pred=pred_y, pos_label=0)
    neg_f1 = f1_score(y_true=true_y, y_pred=pred_y, pos_label=0)
    confusion = confusion_matrix(y_true=true_y, y_pred=pred_y)
    return {'pos_pre':pos_pre,'pos_rec':pos_rec,'pos_f1':pos_f1,
            'neg_pre':neg_pre,'neg_rec':neg_rec,'neg_f1':neg_f1,
            'confusion':confusion}


def collate(sample_list, tokenizer):
    pad_batch_text = {
        'input_ids':[x['text_input']['input_ids'] for x in sample_list],
        'attention_mask':[x['text_input']['attention_mask'] for x in sample_list],
    }
    pad_batch_text = tokenizer.pad(pad_batch_text, return_tensors='pt')
    s_inputs = pad_batch_text['input_ids']
    s_attention_mask = pad_batch_text['attention_mask']
    
    batch_label = torch.tensor([x['label'] for x in sample_list], dtype=torch.long)
    batch_name = [x['audio_name'] for x in sample_list]
    return ((s_inputs, s_attention_mask), batch_label, batch_name)


if __name__ == '__main__':
    
    data_source = ['/mnt/data/GodEye_Emotion/dahai-online_meta.csv',
                   '/mnt/data/GodEye_Emotion/qingqing-online_meta.csv',
                   '/mnt/data/GodEye_Emotion/zhikang-online_meta.csv',
                   '/mnt/data/GodEye_Emotion/peiyou-online_meta.csv']
    
    data_root = '/mnt/data/GodEye_Emotion/'

    # Here we need to process with the train/valid/test

    train_data, valid_data, test_data = [], [], []
    for data_path in data_source:
        data_meta = pd.read_csv(data_path)
        train_data.extend([(os.path.join(data_root,x['wav_path']),x['asr_text'],x['label']) for x in data_meta[data_meta['set']=='train'].to_dict('records')])
        valid_data.extend([(os.path.join(data_root,x['wav_path']),x['asr_text'],x['label']) for x in data_meta[data_meta['set']=='valid'].to_dict('records')])
        test_data.extend([(os.path.join(data_root,x['wav_path']),x['asr_text'],x['label']) for x in data_meta[data_meta['set']=='test'].to_dict('records')])
        
    tokenizer = BertTokenizer.from_pretrained(
        '/share/small_project/auto_text_classifier/atc/data/chinese_roberta_wwm_ext')
    model = BertForSequenceClassification.from_pretrained(
        '/share/small_project/auto_text_classifier/atc/data/chinese_roberta_wwm_ext')
    model.cuda()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    epoch_num = 20
    batch_size = 32
    num_workers = 4

    train_dataset = PassionDataset(train_data, tokenizer)
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset, batch_size = batch_size, 
        collate_fn=lambda x: collate(x, tokenizer),
        shuffle = True, num_workers = num_workers
    )

    valid_dataset = PassionDataset(valid_data, tokenizer)
    valid_loader = torch.utils.data.DataLoader(
        dataset = valid_dataset, batch_size = batch_size, 
        collate_fn=lambda x: collate(x, tokenizer),
        shuffle = True, num_workers = num_workers
    )

    test_dataset = PassionDataset(test_data, tokenizer)
    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset, batch_size = batch_size, 
        collate_fn=lambda x: collate(x, tokenizer),
        shuffle = True, num_workers = num_workers
    )

    best_f1, save_metric = 0, None
    for epoch in range(epoch_num):
        epoch_train_loss = []
        model.train()
        start_time = time.time()

        for semantic_inputs, label_inputs, _ in train_loader:
            input_ids = semantic_inputs[0].cuda()
            attention_mask = semantic_inputs[1].cuda()
            labels = label_inputs.cuda()
            
            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            epoch_train_loss.append(loss)
            loss.backward()
            optimizer.step()

        epoch_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()

        pred_y, true_y = [], []
        model.eval()

        for semantic_inputs, label_inputs, _ in valid_loader:
            input_ids = semantic_inputs[0].cuda()
            attention_mask = semantic_inputs[1].cuda()
            true_y.extend(label_inputs.numpy())
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=None)
            label_outputs = torch.argmax(outputs.logits,axis=1).cpu().detach().numpy().astype(int)
            pred_y.extend(list(label_outputs))

        pred_y = np.array(pred_y)
        true_y = np.array(true_y)

        metric = metric_helper(pred_y, true_y)
        avg_f1 = (metric['pos_f1']+metric['neg_f1'])/2

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("Avg_f1: {:.3f} - Train Loss: {:.3f}".format(avg_f1, epoch_train_loss))

        if avg_f1 > best_f1:
            
            best_f1 = avg_f1
            pred_y, true_y = [], []
            
            for semantic_inputs, label_inputs, _ in  test_loader:
                input_ids = semantic_inputs[0].cuda()
                attention_mask = semantic_inputs[1].cuda()
                true_y.extend(label_inputs.numpy())
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=None)
                label_outputs = torch.argmax(outputs.logits,axis=1).cpu().detach().numpy().astype(int)
                pred_y.extend(list(label_outputs))

                metric = metric_helper(pred_y, true_y)
                save_metric = metric

    save_path = '/home/work/Projects/GodEye/result/roberta_report.pkl'
    pickle.dump(save_metric, open(save_path,'wb'))
    print('Work finished! Best result saved to {}'.format(save_path))