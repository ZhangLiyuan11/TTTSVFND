import os
import pickle

import h5py
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset
from tools import *
import random
def str2num(str_x):
    # isinstance判断一个对象是不是一个已知类型
    if isinstance(str_x, float):
        return str_x
    # isdigit 检测字符串是否只由数字组成
    elif str_x.isdigit():
        return int(str_x)
    elif 'w' in str_x:
        return float(str_x[:-1])*10000
    elif '亿' in str_x:
        return float(str_x[:-1])*100000000
    else:
        print ("error")
        print (str_x)


class T3SVFNDDataset(Dataset):
    def __init__(self, path_vid, mask_ratio, tokenizer,datamode='title+ocr'):
        self.mask_ratio = mask_ratio
        self.data_complete = pd.read_json('dataset\data_complete.json', orient='records', dtype=False, lines=True)

        # self.data_complete = self.data_complete[self.data_complete['label']!=2] # label: 0-real, 1-fake, 2-debunk
        self.data_complete = self.data_complete[self.data_complete['annotation'] != '辟谣']
        self.framefeapath='dataset\ptvgg19_frames'
        self.c3dfeapath = 'dataset/c3d/'
        self.comments_bert_fea_path = 'dataset\comments_bert_em/'
        self.hubert_path = 'dataset\hubert_ems\\'
        self.vid = []
        
        with open('dataset\\vids\\'+path_vid, "r") as fr:
            for line in fr.readlines():
                self.vid.append(line.strip())

        self.data = self.data_complete[self.data_complete.video_id.isin(self.vid)]
        # 设置类标签
        self.data['video_id'] = self.data['video_id'].astype('category')
        # 改变标签类别
        self.data['video_id'].cat.set_categories(self.vid)
        # 按照类别排序，而不是字母排序
        self.data.sort_values('video_id', ascending=True, inplace=True)
        # 重置数据帧的索引，并使用默认索引
        self.data.reset_index(inplace=True)  

        self.tokenizer = tokenizer
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        self.datamode = datamode
        
    def __len__(self):
        return self.data.shape[0]
     
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # label 
        label = 0 if item['annotation']=='真' else 1
        label = torch.tensor(label)

        # text
        if self.datamode == 'title+ocr':
            title_tokens = self.tokenizer(item['title']+' '+item['ocr'], max_length=512, padding='max_length', truncation=True)
        elif self.datamode == 'ocr':
            title_tokens = self.tokenizer(item['ocr'], max_length=512, padding='max_length', truncation=True)
        elif self.datamode == 'title':
            title_tokens = self.tokenizer(item['title'], max_length=512, padding='max_length', truncation=True)
        title_inputid = torch.LongTensor(title_tokens['input_ids'])
        title_mask = torch.LongTensor(title_tokens['attention_mask'])

        mask_token = 103
        num_to_mask = int(self.mask_ratio * torch.sum(title_mask))
        # 随机选择要遮蔽的位置,仅遮蔽有效词元
        mask_indices = random.sample(range(torch.sum(title_mask)), num_to_mask)
        # 进行遮蔽操作
        title_inputid[mask_indices] = mask_token
        title_mask[mask_indices] = 0

        # 创建标签张量，初始值为-100
        auxiliary_labels = torch.full_like(title_inputid, -100)
        # 设置被遮蔽位置的标签为原始值
        auxiliary_labels[mask_indices] = title_inputid[mask_indices]

        audio_item_path = self.hubert_path + vid + '.pkl'
        audio_fea = torch.load(audio_item_path)

        # path = os.path.join(self.framefeapath,vid+'.pkl')
        # frames
        frames=pickle.load(open(os.path.join(self.framefeapath,vid+'.pkl'),'rb'))
        frames=torch.FloatTensor(frames)

        # comments
        comments_fea = torch.load(open(os.path.join(self.comments_bert_fea_path, vid + '.pkl'), 'rb'))

        # # user
        try: 
            if item['is_author_verified'] == 1:
                intro = "个人认证"
            elif item['is_author_verified'] == 2:
                intro = "机构认证"
            elif item['is_author_verified'] == 0:
                intro = "未认证"
            else: 
                intro = "认证状态未知"
        except:
            if 'author_verified_intro' == '':
                intro = "认证状态未知"
            else:
                intro = "有认证"

        for key in ['author_intro', 'author_verified_intro']:
            try:
                intro = intro + ' ' + item[key]
            except:
                intro += ' '

        intro_tokens = self.tokenizer(intro, max_length=50, padding='max_length', truncation=True)
        intro_inputid = torch.LongTensor(intro_tokens['input_ids'])
        intro_mask = torch.LongTensor(intro_tokens['attention_mask'])

        # video
        c3d = h5py.File(self.c3dfeapath + vid + ".hdf5", "r")[vid]['c3d_features']
        c3d = torch.FloatTensor(c3d)

        return {
            'label': label,
            'title_inputid': title_inputid,
            'title_mask': title_mask,
            'audio_fea': audio_fea,
            'frames':frames,
            'intro_inputid': intro_inputid,
            'intro_mask': intro_mask,
            'c3d': c3d,
            'comments_fea': comments_fea,
            'auxiliary_labels': auxiliary_labels,
        }


def pad_frame_sequence(seq_len,lst):
    attention_masks = []
    result=[]
    for video in lst:
        video=torch.FloatTensor(video)
        ori_len=video.shape[0]
        if ori_len>=seq_len:
            gap=ori_len//seq_len
            video=video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video=torch.cat((video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.float)),dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len-ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)

def pad_audio_sequence(seq_len,lst):
    attention_masks = []
    result=[]
    for video in lst:
        video = torch.squeeze(video)
        # video=torch.FloatTensor(video)
        ori_len=video.shape[0]
        if ori_len>=seq_len:
            gap=ori_len//seq_len
            video=video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video=torch.cat((video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.float)),dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len-ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)

def collate_fn(batch):
    num_frames = 83
    num_audioframes = 50

    auxiliary_labels = [item['auxiliary_labels'] for item in batch]

    intro_inputid = [item['intro_inputid'] for item in batch]
    intro_mask = [item['intro_mask'] for item in batch]

    title_inputid = [item['title_inputid'] for item in batch]
    title_mask = [item['title_mask'] for item in batch]

    comments_feas = [item['comments_fea'] for item in batch]

    # 根据帧数补齐关键帧特征
    frames = [item['frames'] for item in batch]
    frames, frames_masks = pad_frame_sequence(num_frames, frames)

    # 根据语音帧数补齐语音特征
    audio_feas = [item['audio_fea'] for item in batch]
    audio_feas, audiofeas_masks = pad_audio_sequence(num_audioframes, audio_feas)

    c3d = [item['c3d'] for item in batch]
    c3d, c3d_masks = pad_frame_sequence(num_frames, c3d)

    label = [item['label'] for item in batch]

    return {
        'label': torch.stack(label),
        'intro_inputid': torch.stack(intro_inputid),
        'intro_mask': torch.stack(intro_mask),
        'title_inputid': torch.stack(title_inputid),
        'title_mask': torch.stack(title_mask),
        'audio_feas': audio_feas,
        'audiofeas_masks': audiofeas_masks,
        'frames': frames,
        'frames_masks': frames_masks,
        'comments_feas': torch.stack(comments_feas),
        'c3d': c3d,
        'c3d_masks': c3d_masks,
        'auxiliary_labels': torch.stack(auxiliary_labels),
    }

