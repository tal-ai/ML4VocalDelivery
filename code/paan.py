import os
import re
import time
import pickle
import librosa
import numpy as np
import pandas as pd

from models import Cnn14
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
# import torch.optim as optim
# import torch.utils.data
from transformers import AdamW, BertTokenizer


class PassionDataset(object):
    def __init__(self, data_list, tokenizer, sample_rate, audio_length=None):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.audio_length = audio_length

    def __len__(self,):
        return len(self.data_list)

    def __getitem__(self, index):
        audio_path, asr_text, label = self.data_list[index]
        audio_name = os.path.basename(audio_path)
        waveform, _ = librosa.core.load(audio_path, sr=self.sample_rate, mono=True)
        if self.audio_length is not None:
            audio_frames = self.sample_rate * self.audio_length
            if len(waveform) < audio_frames:
                waveform = np.concatenate((waveform, np.zeros(audio_frames - len(waveform))), axis=0)
            else:
                waveform = waveform[0:audio_frames]
        return {'waveform':waveform,'label':label,'audio_name':audio_name}


def metric_helper(pred_label, true_label):
    acc = accuracy_score(y_true = true_label, y_pred = pred_label)
    pos_pre = precision_score(y_true = true_label, y_pred = pred_label, pos_label=1)
    neg_pre = precision_score(y_true = true_label, y_pred = pred_label, pos_label=0)
    pos_rec = recall_score(y_true = true_label, y_pred = pred_label, pos_label=1)
    neg_rec = recall_score(y_true = true_label, y_pred = pred_label, pos_label=0)
    pos_f1 = f1_score(y_true = true_label, y_pred = pred_label, pos_label=1)
    neg_f1 = f1_score(y_true = true_label, y_pred = pred_label, pos_label=0)

    key_metric = (pos_f1 + neg_f1) / 2
    report_metric = {'acc':acc,'pos_pre':pos_pre,'pos_rec':pos_rec,'pos_f1':pos_f1,
                     'neg_pre':neg_pre,'neg_rec':neg_rec,'neg_f1':neg_f1}

    return key_metric, report_metric


def collate(sample_list, tokenizer):
    batch_waveform = torch.FloatTensor([x['waveform'] for x in sample_list])
    batch_label = torch.LongTensor([x['label'] for x in sample_list])
    batch_name = [x['audio_name'] for x in sample_list]
    return batch_waveform, batch_label, batch_name


class PAAN(nn.Module):
    def __init__(self, classes_num=2):
        super().__init__()
        sample_rate, window_size, hop_size, mel_bins, fmin, fmax = 16000, 512, 160, 64, 50, 8000
        self.base = Cnn14(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, 527)
        checkpoint = torch.load('/workspace/HangLi/class_rehearsal/emo/PAAN/pretrain_model/Cnn14_16k_mAP=0.438.pth')
        self.base.load_state_dict(checkpoint['model'])
        self.classifier = nn.Linear(2048, classes_num)

    def forward(self, waveform_inputs):
        acoustic_outputs = self.base(waveform_inputs, None)
        acoustic_encode = acoustic_outputs['embedding']
        logits = self.classifier(acoustic_encode)
        return logits


if __name__ == '__main__':
    
    # data_source = ['/mnt/data/GodEye_Emotion/dahai-online_meta.csv',
    #                '/mnt/data/GodEye_Emotion/qingqing-online_meta.csv',
    #                '/mnt/data/GodEye_Emotion/zhikang-online_meta.csv',
    #                '/mnt/data/GodEye_Emotion/peiyou-online_meta.csv']
    
    data_source = ['/mnt/data/GodEye_Emotion/dada-online_meta.csv',]
    
    data_root = '/mnt/data/GodEye_Emotion/'

    # Here we need to process with the train/valid/test

    train_data, valid_data, test_data = [], [], []
    for data_path in data_source:
        data_meta = pd.read_csv(data_path)
        # train_data.extend([(os.path.join(data_root,x['wav_path']),x['asr_text'],x['label']) for x in data_meta[data_meta['set']=='train'].to_dict('records')])
        # valid_data.extend([(os.path.join(data_root,x['wav_path']),x['asr_text'],x['label']) for x in data_meta[data_meta['set']=='valid'].to_dict('records')])
        # test_data.extend([(os.path.join(data_root,x['wav_path']),x['asr_text'],x['label']) for x in data_meta[data_meta['set']=='test'].to_dict('records')])

        train_data.extend([(os.path.join(data_root,x['wav_path']),x['human_text'],x['label']) for x in data_meta.to_dict('records') if x['set']=='train' and x['gender']=='male'])
        valid_data.extend([(os.path.join(data_root,x['wav_path']),x['human_text'],x['label']) for x in data_meta.to_dict('records') if x['set']=='valid' and x['gender']=='male'])
        test_data.extend([(os.path.join(data_root,x['wav_path']),x['human_text'],x['label']) for x in data_meta.to_dict('records') if x['set']=='test' and x['gender']=='male'])

    # tokenizer = BertTokenizer.from_pretrained('/share/small_project/auto_text_classifier/atc/data/chinese_roberta_wwm_ext')
    tokenizer = None
    
    model = PAAN(2)
    model.cuda()

    # no_decay = ['bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    # ]
    # optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, 
        betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    epoch_num = 20
    batch_size = 32
    num_workers = 4
    sample_rate = 16000
    audio_length = 10
    loss_function = nn.CrossEntropyLoss()


    train_dataset = PassionDataset(train_data, tokenizer, sample_rate, audio_length)
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset, batch_size = batch_size, 
        collate_fn=lambda x: collate(x, tokenizer),
        shuffle = True, num_workers = num_workers
    )

    valid_dataset = PassionDataset(valid_data, tokenizer, sample_rate, audio_length)
    valid_loader = torch.utils.data.DataLoader(
        dataset = valid_dataset, batch_size = batch_size, 
        collate_fn=lambda x: collate(x, tokenizer),
        shuffle = True, num_workers = num_workers
    )

    test_dataset = PassionDataset(test_data, tokenizer, sample_rate, audio_length)
    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset, batch_size = batch_size, 
        collate_fn=lambda x: collate(x, tokenizer),
        shuffle = True, num_workers = num_workers
    )

    best_metric, save_metric = 0, None
    for epoch in range(epoch_num):
        epoch_train_loss = []
        model.train()
        start_time = time.time()

        for acoustic_inputs, label_inputs, _ in train_loader:
            waveform_inputs = acoustic_inputs.cuda()
            labels = label_inputs.cuda()
            
            model.zero_grad()
            prediction = model(waveform_inputs)
            
            loss = loss_function(prediction, labels)
            epoch_train_loss.append(loss)

            loss.backward()
            optimizer.step()
        
        epoch_train_loss = torch.mean(torch.tensor(epoch_train_loss)).cpu().detach().numpy()
        
        pred_y, true_y = [], []
        model.eval()

        for acoustic_inputs, label_inputs, _ in valid_loader:
            waveform_inputs = acoustic_inputs.cuda()
            labels = label_inputs.cuda()
            true_y.extend(label_inputs.numpy())

            prediction = model(waveform_inputs)
            label_outputs = torch.argmax(prediction,axis=1).cpu().detach().numpy().astype(int)
            pred_y.extend(list(label_outputs))

        pred_y = np.array(pred_y)
        true_y = np.array(true_y)

        key_metric, report_metric = metric_helper(pred_y, true_y)

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
                time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print('Valid Metric: {} - Train Loss: {:.3f}'.format(
              ' - '.join(['{}: {:.3f}'.format(key, value) for key, value in report_metric.items()]),
              epoch_train_loss))

        if key_metric > best_metric:
            print('Better Metric found on dev, calculate performance on Test')
            best_metric = key_metric
            pred_y, true_y = [], []
            
            for acoustic_inputs, label_inputs, _ in test_loader:
                waveform_inputs = acoustic_inputs.cuda()
                labels = label_inputs.cuda()
                true_y.extend(label_inputs.numpy())

                prediction = model(waveform_inputs)
                label_outputs = torch.argmax(prediction,axis=1).cpu().detach().numpy().astype(int)
                pred_y.extend(list(label_outputs))
                
                key_metric, report_metric = metric_helper(pred_y, true_y)
                save_metric = report_metric

            print("Test Metric: {}".format(
                ' - '.join(['{}: {:.3f}'.format(key, value) for key, value in save_metric.items()])
            ))

    # save_path = '/home/work/Projects/GodEye/result/paan_report.pkl'
    save_path = '/home/work/Projects/GodEye/result/dada-paan_report.pkl'
    pickle.dump(save_metric, open(save_path,'wb'))
    print('Work finished! Best result saved to {}'.format(save_path))