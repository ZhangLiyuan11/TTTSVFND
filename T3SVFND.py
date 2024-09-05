from transformers import BertModel
import torch
from torch import nn

from trans_model import *
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from tools import *
loss_function_auxiliary = nn.CosineEmbeddingLoss(reduction='mean')
loss_ce = nn.CrossEntropyLoss(ignore_index=-100)


class auxiliary_model(torch.nn.Module):
    def __init__(self,fea_dim, dropout):
        super(auxiliary_model, self).__init__()
        # 加载bert模型
        # self.bert = bert_model

        # 维度
        self.text_dim = 1024
        self.img_dim = 4096
        self.comment_dim = 1024
        self.video_dim = 4096
        self.num_frames = 83
        self.num_audioframes = 50
        self.num_comments = 23
        self.dim = fea_dim
        self.num_heads = 8
        self.trans_dim = 512
        self.hubert_dim = 1024
        self.dropout = dropout

        # 语义交互共注意力模块
        self.at_MCT = Transformer(model_dimension=self.trans_dim, number_of_heads=self.num_heads,
                                  dropout_probability=self.dropout)
        self.tv_MCT = Transformer(model_dimension=self.trans_dim, number_of_heads=self.num_heads,
                                  dropout_probability=self.dropout)

        self.decoder_a = nn.TransformerEncoderLayer(d_model=self.trans_dim, nhead=2, batch_first=True)
        self.decoder_v = nn.TransformerEncoderLayer(d_model=self.trans_dim, nhead=2, batch_first=True)
        self.token_class_at = nn.Linear(self.trans_dim, 21128)
        self.token_class_vt = nn.Linear(self.trans_dim, 21128)

    def forward(self,fea_text,fea_img,fea_audio,**kwargs):
        # 辅助任务的标签
        auxiliary_labels = kwargs['auxiliary_labels']

        # Mask语义交互
        fea_ta = self.at_MCT(fea_text, fea_audio)
        fea_tv = self.tv_MCT(fea_text, fea_img)

        # MAE_text_decode重建
        audio_decode_fea = self.decoder_a(fea_ta)
        video_decode_fea = self.decoder_v(fea_tv)
        pre_vt = self.token_class_vt(video_decode_fea)
        pre_at = self.token_class_at(audio_decode_fea)
        loss_a = loss_ce(pre_at.view(-1, 21128), auxiliary_labels.view(-1))
        loss_v = loss_ce(pre_vt.view(-1, 21128), auxiliary_labels.view(-1))

        return loss_a, loss_v, fea_ta, fea_tv

class project_module(torch.nn.Module):
    def __init__(self,fea_dim, dropout):
        super(project_module, self).__init__()
        # 维度
        self.text_dim = 1024
        self.img_dim = 4096
        self.dim = fea_dim
        self.trans_dim = 512
        self.hubert_dim = 1024
        self.dropout = dropout

        self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, self.trans_dim), torch.nn.ReLU(),
                                         nn.Dropout(p=self.dropout))
        self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, self.trans_dim), torch.nn.ReLU(),
                                        nn.Dropout(p=self.dropout))
        self.linear_hubert = nn.Sequential(torch.nn.Linear(self.hubert_dim, self.trans_dim), torch.nn.ReLU(),
                                           nn.Dropout(p=self.dropout))


    def forward(self,fea_text,**kwargs):
        ### Title ###
        fea_text = self.linear_text(fea_text)

        ### Audio Frames ###
        audio_feas = kwargs['audio_feas']
        fea_audio = self.linear_hubert(audio_feas)

        ### Image Frames ###
        frames = kwargs['frames']
        fea_img = self.linear_img(frames)

        return fea_text, fea_audio, fea_img


class T3SVFENDModel(torch.nn.Module):
    def __init__(self, fea_dim, dropout):
        super(T3SVFENDModel, self).__init__()

        # 维度
        self.text_dim = 1024
        self.img_dim = 4096
        self.comment_dim = 1024
        self.video_dim = 4096
        self.num_frames = 83
        self.num_audioframes = 50
        self.num_comments = 23
        self.dim = fea_dim
        self.num_heads = 8
        self.trans_dim = 512
        self.hubert_dim = 1024

        self.dropout = dropout

        self.linear_comment = nn.Sequential(torch.nn.Linear(self.comment_dim, self.trans_dim), torch.nn.ReLU(),
                                            nn.Dropout(p=self.dropout))
        self.linear_video = nn.Sequential(torch.nn.Linear(self.video_dim, self.trans_dim), torch.nn.ReLU(),
                                          nn.Dropout(p=self.dropout))
        self.linear_intro = nn.Sequential(torch.nn.Linear(self.text_dim, self.trans_dim), torch.nn.ReLU(),
                                          nn.Dropout(p=self.dropout))

        self.vt_CT = Transformer(model_dimension=self.trans_dim, number_of_heads=self.num_heads,
                                  dropout_probability=self.dropout)

        # 最终的融合模块
        self.trm = nn.TransformerEncoderLayer(d_model=self.trans_dim, nhead=2, batch_first=True)
        self.classifier = nn.Linear(self.trans_dim, 2)


    def forward(self,fea_text,fea_img,fea_ta,fea_tv,fea_intro,**kwargs):
        # tv语义交互
        fea_vt = self.vt_CT(fea_img,fea_text)
        fea_vt = torch.mean(fea_vt, -2)

        fea_ta = torch.mean(fea_ta, -2)
        fea_tv = torch.mean(fea_tv, -2)

        ### C3D ###
        c3d = kwargs['c3d']  # (batch, 36, 4096)
        fea_video = self.linear_video(c3d)  # (batch, frames, 128)
        fea_video = torch.mean(fea_video, -2)

        ### Comment ###
        comments_feas = kwargs['comments_feas']  # (batch,1024)
        fea_comments = self.linear_comment(comments_feas)  # (batch,fea_dim)

        fea_intro = self.linear_intro(fea_intro)

        # 特征融合
        fea_ta = fea_ta.unsqueeze(1)
        fea_tv = fea_tv.unsqueeze(1)
        fea_vt = fea_vt.unsqueeze(1)
        fea_comments = fea_comments.unsqueeze(1)
        fea_video = fea_video.unsqueeze(1)
        fea_intro = fea_intro.unsqueeze(1)
        # fea_t = fea_t.unsqueeze(1)
        fea = torch.cat((fea_ta,fea_tv,fea_vt,fea_intro, fea_video, fea_comments), 1)  # (bs, 5, 128)
        fea = self.trm(fea)
        fea = torch.mean(fea, -2)

        # final_fea = torch.cat((fea_at,fea_vt,fea_tv),dim=1)
        output = self.classifier(fea)

        return output
