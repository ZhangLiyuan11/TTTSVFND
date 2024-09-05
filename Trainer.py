import copy
import os
import time

import tqdm
from tqdm import tqdm
from metrics import *
from zmq import device
from tools import *

class Trainer():
    def __init__(self,
                model,
                 auxiliary_model,
                 project_model,
                 device,
                 lr,
                 dropout,
                 dataloaders,
                 alpha,
                 weight_decay,
                 save_param_path,
                 writer, 
                 epoch_stop,
                 epoches,
                 mode,
                 model_name, 
                 event_num,
                 save_threshold = 0.0, 
                 start_epoch = 0,
                 ):
        
        self.auxiliary_model = auxiliary_model
        self.project_model = project_model
        self.model = model
        self.alpha = alpha
        self.device = device
        self.mode = mode
        self.model_name = model_name
        self.event_num = event_num
        self.bert_model = pretrain_bert_models()
        self.dataloaders = dataloaders
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.epoch_stop = epoch_stop
        self.save_threshold = save_threshold
        self.writer = writer

        if os.path.exists(save_param_path):
            self.save_param_path = save_param_path
        else:
            self.save_param_path = os.makedirs(save_param_path)
            self.save_param_path= save_param_path

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
    
        self.criterion = nn.CrossEntropyLoss()
        

    def train(self,dataloader_list):
        since = time.time()
        self.model.cuda()
        self.auxiliary_model.cuda()
        self.project_model.cuda()
        best_acc_test = 0.0
        best_epoch_test = 0

        is_earlystop = False

        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch+1, self.start_epoch+self.num_epochs))
            print('-' * 50)

            # 更新学习率
            p = float(epoch) / 100
            lr = self.lr / (1. + 10 * p) ** 0.75
            # 创建优化器
            self.auxiliary_optimizer = torch.optim.Adam(params=self.auxiliary_model.parameters(), lr=lr)
            self.model_optimizer = torch.optim.Adam([
                {'params': self.model.parameters()},
                {'params': self.project_model.parameters()}
            ], lr=lr)

            for phase in dataloader_list:
                if phase == 'train':
                    self.model.train()
                    self.project_model.train()
                    self.auxiliary_model.train()
                elif phase == 'test_time_train':
                    self.model.eval()
                    self.project_model.eval()
                    self.auxiliary_model.train()
                else:
                    self.project_model.eval()
                    self.model.eval()
                    self.auxiliary_model.eval()

                print('-' * 10)
                print (phase.upper())
                print('-' * 10)

                running_loss = 0.0
                loss_auxiliary = 0.0
                tpred = []
                tlabel = []

                for batch in tqdm(self.dataloaders[phase]):
                    batch_data=batch
                    for k,v in batch_data.items():
                        batch_data[k]=v.cuda()
                    # pre-trained model
                    with torch.no_grad():
                        ### text ###
                        fea_text = self.bert_model(batch_data['title_inputid'], attention_mask=batch_data['title_mask'])['last_hidden_state']
                        ### User Intro ###
                        fea_intro = self.bert_model(batch_data['intro_inputid'], attention_mask=batch_data['intro_mask'])[1]

                    label = batch_data['label']

                    if phase == 'train':
                        with torch.set_grad_enabled(True):
                            fea_text, fea_audio, fea_img = self.project_model(fea_text,**batch_data)
                            loss_txt, loss_audio, fea_ta, fea_tv = self.auxiliary_model(fea_text,fea_img,fea_audio,**batch_data)
                            outputs = self.model(fea_text,fea_img,fea_ta,fea_tv,fea_intro,**batch_data)
                            _, preds = torch.max(outputs, 1)
                            loss = self.criterion(outputs, label)
                            loss_final = loss + self.alpha * (loss_txt + loss_audio)

                            self.model_optimizer.zero_grad()
                            self.auxiliary_optimizer.zero_grad()
                            loss_final.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.model_optimizer.step()

                            torch.nn.utils.clip_grad_norm_(self.auxiliary_model.parameters(), 1.0)
                            self.auxiliary_optimizer.step()

                    if phase == 'test_time_train' :
                        with torch.set_grad_enabled(True):
                            fea_text, fea_audio, fea_img = self.project_model(fea_text, **batch_data)
                            loss_txt, loss_audio, fea_ta, fea_tv = self.auxiliary_model(fea_text, fea_img, fea_audio,
                                                                                        **batch_data)
                            outputs = self.model(fea_text, fea_img, fea_ta, fea_tv, fea_intro, **batch_data)
                            _, preds = torch.max(outputs, 1)
                            loss = self.criterion(outputs, label)
                            loss_final = loss_txt + loss_audio

                            self.auxiliary_optimizer.zero_grad()
                            loss_final.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.auxiliary_optimizer.step()

                    if phase == 'test' or phase == 'val':
                        with torch.set_grad_enabled(False):
                            fea_text, fea_audio, fea_img = self.project_model(fea_text, **batch_data)
                            loss_txt, loss_audio, fea_ta, fea_tv = self.auxiliary_model(fea_text, fea_img, fea_audio,
                                                                                        **batch_data)
                            outputs = self.model(fea_text, fea_img, fea_ta, fea_tv, fea_intro, **batch_data)
                            _, preds = torch.max(outputs, 1)
                            loss = self.criterion(outputs, label)
                    tlabel.extend(label.detach().cpu().numpy().tolist())
                    tpred.extend(preds.detach().cpu().numpy().tolist())
                    running_loss += loss.item() * label.size(0)
                    loss_auxiliary += (loss_txt.item() + loss_audio.item()) * label.size(0)

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_loss_auxiliary = loss_auxiliary / len(self.dataloaders[phase].dataset)
                print('Loss_auxiliary: {:.4f} '.format(epoch_loss_auxiliary))
                print('Loss: {:.4f} '.format(epoch_loss))
                results = metrics(tlabel, tpred)
                print (results)
                get_confusionmatrix_fnd(tpred,tlabel)
                self.writer.add_scalar('Loss/'+phase, epoch_loss, epoch+1)
                self.writer.add_scalar('Acc/'+phase, results['acc'], epoch+1)
                self.writer.add_scalar('F1/'+phase, results['f1'], epoch+1)

                if phase == 'test' and results['acc'] > best_acc_test:
                    best_acc_test = results['acc']
                    best_epoch_test = epoch+1
                    if best_acc_test > self.save_threshold:
                        torch.save(self.model.state_dict(), self.save_param_path + "_test_epoch" + str(best_epoch_test) + "_{0:.4f}".format(best_acc_test))
                        print ("saved " + self.save_param_path + "_test_epoch" + str(best_epoch_test) + "_{0:.4f}".format(best_acc_test) )
                    else:
                        if epoch-best_epoch_test >= self.epoch_stop-1:
                            is_earlystop = True
                            print ("early stopping...")
                torch.cuda.empty_cache()
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best model on val: epoch" + str(best_epoch_test) + "_" + str(best_acc_test))
        return True
