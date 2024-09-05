from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from T3SVFND import T3SVFENDModel,auxiliary_model,project_module

from dataloader import *
from Trainer import *

def _init_fn(worker_id):
    np.random.seed(2022)

class Run():
    def __init__(self,config):
        self.model_name = config['model_name']
        self.mode_eval = config['mode_eval']
        self.fold = config['fold']
        self.data_type = 'T3SVFND'
        self.epoches = config['epoches']
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.epoch_stop = config['epoch_stop']
        self.seed = config['seed']
        self.device = config['device']
        self.lr = config['lr']
        self.lambd=config['lambd']
        self.save_param_dir = config['path_param']
        self.path_tensorboard = config['path_tensorboard']
        self.dropout = config['dropout']
        self.weight_decay = config['weight_decay']
        self.mode ='normal'
        self.mask_radio = config['mask_radio']
        self.alpha = config['alpha']

    def get_dataloader_temporal(self, data_type,mask_radio):
        tokenizer = pretrain_bert_token()
        dataset_train = T3SVFNDDataset('vid_time3_train.txt', mask_ratio=mask_radio,tokenizer=tokenizer)
        dataset_test_time_train = T3SVFNDDataset('vid_time3_test.txt', mask_ratio=mask_radio,tokenizer=tokenizer)
        dataset_val = T3SVFNDDataset('vid_time3_val.txt', mask_ratio=0,tokenizer=tokenizer)
        dataset_test = T3SVFNDDataset('vid_time3_test.txt', mask_ratio=0,tokenizer=tokenizer)

        train_dataloader = DataLoader(dataset_train, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn)
        val_dataloader = DataLoader(dataset_val, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn)
        test_time_train_dataloader=DataLoader(dataset_test_time_train, batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            worker_init_fn=_init_fn,
            collate_fn=collate_fn)
        test_dataloader = DataLoader(dataset_test, batch_size=self.batch_size,
                                                num_workers=self.num_workers,
                                                pin_memory=True,
                                                shuffle=False,
                                                worker_init_fn=_init_fn,
                                                collate_fn=collate_fn)
 
        dataloaders =  dict(zip(['train', 'val', 'test_time_train','test'],[train_dataloader, val_dataloader,test_time_train_dataloader, test_dataloader]))
        return dataloaders

    def get_dataloader_nocv(self, data_type,data_fold,mask_radio):
        tokenizer = pretrain_bert_token()
        dataset_train = T3SVFNDDataset(f'vid_fold_no_{data_fold}.txt', mask_ratio=mask_radio,tokenizer=tokenizer)
        dataset_test_time_train = T3SVFNDDataset(f'vid_fold_{data_fold}.txt', mask_ratio=mask_radio,tokenizer=tokenizer)
        dataset_test = T3SVFNDDataset(f'vid_fold_{data_fold}.txt', mask_ratio=0,tokenizer=tokenizer)

        train_dataloader = DataLoader(dataset_train, batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      pin_memory=True,
                                      shuffle=True,
                                      worker_init_fn=_init_fn,
                                      collate_fn=collate_fn)
        test_time_train_dataloader = DataLoader(dataset_test_time_train, batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     pin_memory=True,
                                     shuffle=False,
                                     worker_init_fn=_init_fn,
                                     collate_fn=collate_fn)
        test_dataloader = DataLoader(dataset_test, batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     pin_memory=True,
                                     shuffle=False,
                                     worker_init_fn=_init_fn,
                                     collate_fn=collate_fn)
        dataloaders = dict(zip(['train','test_time_train','test'], [train_dataloader,test_time_train_dataloader,test_dataloader]))
        return dataloaders

    def get_model(self):
        if self.model_name == 'T3SVFEND':
            self.model = T3SVFENDModel(fea_dim=128,dropout=self.dropout)
            self.auxiliary_model = auxiliary_model(fea_dim=128,dropout=self.dropout)
            self.project_model = project_module(fea_dim=128,dropout=self.dropout)
        return self.model,self.auxiliary_model,self.project_model

    def main(self):
        self.model,self.auxiliary_model,self.project_model = self.get_model()
        if self.mode_eval == 'temporal':
            dataloaders = self.get_dataloader_temporal(data_type=self.data_type,mask_radio=self.mask_radio)
            trainer = Trainer(model=self.model, auxiliary_model=self.auxiliary_model,project_model=self.project_model, device=self.device, lr=self.lr, dataloaders=dataloaders,alpha=self.alpha,
                               epoches=self.epoches, dropout=self.dropout, weight_decay=self.weight_decay,
                               mode=self.mode, model_name=self.model_name, event_num=self.event_num,
                               epoch_stop=self.epoch_stop,
                               save_param_path=self.save_param_dir + self.data_type + "/" + self.model_name + "/",
                               writer=SummaryWriter(self.path_tensorboard))
            dataloader_list = ['train', 'test_time_train', 'val', 'test']
            result = trainer.train(dataloader_list)
            return result
        else:
            dataloaders = self.get_dataloader_nocv(data_type=self.data_type, data_fold=self.fold,mask_radio=self.mask_radio)
            trainer = Trainer(model=self.model,  auxiliary_model=self.auxiliary_model,project_model=self.project_model, device=self.device, lr=self.lr, dataloaders=dataloaders,alpha=self.alpha,
                                    epoches=self.epoches, dropout=self.dropout, weight_decay=self.weight_decay,
                                    mode=self.mode, model_name=self.model_name, event_num=self.event_num,
                                    epoch_stop=self.epoch_stop,
                                    save_param_path=self.save_param_dir + self.data_type + "/" + self.model_name + "/",
                                    writer=SummaryWriter(self.path_tensorboard))
            dataloader_list = ['train','test_time_train','test']
            result = trainer.train(dataloader_list)
        return result
