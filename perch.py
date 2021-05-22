import os
import json
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from settings import data_path,perch_image_path,log_dir,result_dir,output_dir
from datasets import PerchWideDataset
from modules import Net

from sklearn.metrics import roc_auc_score,accuracy_score

import ray
ray.init(address="auto")
# ray.init( num_cpus=12,dashboard_host="0.0.0.0")
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler,HyperBandScheduler,AsyncHyperBandScheduler

from utils import show_grid,save_table3
display=os.environ.get("DISPLAY",None)

input_size=224 #224,512
base_model="resnet34"
experiment=f"Perch-{base_model}"
if input_size == 512: experiment= experiment + f"-{input_size}"


perch_wide=pd.read_csv(os.path.join(perch_image_path,"perch_wide.csv"))
if input_size==512: perch_wide['resized_path']=perch_wide['resized_path'].map(lambda x:x.replace("resized","resized600"))
rng=np.random.RandomState(500)
all_ids = perch_wide['PATID'].unique()
train_ids_ = rng.choice(all_ids, size=int(len(all_ids) * 0.8), replace=False)
train = perch_wide[perch_wide['PATID'].isin(train_ids_)]
test = perch_wide[~perch_wide['PATID'].isin(train_ids_)]





def train_transforms(config):
    return transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(brightness=.1,contrast=.1)],
                               p=config['prop_jitter']),
        transforms.RandomResizedCrop(input_size, ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomAffine(degrees=10,shear=10)],
                               p=config["prop_affine"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

val_transforms=transforms.Compose([
        # transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_dataset=PerchWideDataset(test,label_var='labels',transforms=val_transforms)
test_loader=DataLoader(test_dataset,shuffle=False,batch_size=16,num_workers=5)

if display:
    test_images=PerchWideDataset(test,label_var='labels',
                                 transforms=train_transforms({'prop_jitter':1.0,'prop_affine':1.0}),
                                 # transforms=val_transforms

                                 )
    show_grid(test_images,nrow=5,ncol=5)

def get_loader(config,train_data):
    all_ids = train_data['PATID'].unique()
    train_ids_ = rng.choice(all_ids, size=int(len(all_ids) * 0.8), replace=False)
    train_csv = train_data[train_data['PATID'].isin(train_ids_)]
    val_csv = train_data[~train_data['PATID'].isin(train_ids_)]



    train_ds = PerchWideDataset(train_csv, label_var='labels', transforms=train_transforms(config))
    val_ds = PerchWideDataset(val_csv, label_var='labels', transforms=val_transforms)

    train_loader = DataLoader(train_ds,
                              batch_size=int(config["batch_size"]),
                              shuffle=True, num_workers=5)
    val_loader = DataLoader(val_ds,
                            batch_size=int(config["batch_size"]),
                            shuffle=False, num_workers=5)
    return train_loader,val_loader

def get_model(config):
    model=Net(dropout=config['dropout'])
    return model

def get_optimizer(config,model):
    optimizer=torch.optim.Adam([
        {'params':model.encoder_model.parameters(),},
        {'params':model.fc.parameters(),'weight_decay':config['l2_fc'],'lr':config['lr_fc']},
    ], weight_decay=config['l2'],lr=config['lr'])
    return optimizer

def train_fun(model,optimizer,criterion,device,train_loader,val_loader,scheduler=None):
    train_loss = 0

    model.train()
    # print("epoch: %d >> learning rate at beginning of epoch: %.5f" % (epoch, optimizer.param_groups[0]['lr']))
    for batch_x, batch_y in train_loader:
        batch_x,batch_y = batch_x.to(device, dtype=torch.float),batch_y.to(device)
        logits = model(batch_x)

        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() / len(train_loader)


    model.eval()
    val_loss = 0
    pred_val=[]
    obs_val=[]
    with torch.no_grad():
        for batch_x,batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device, dtype=torch.float), batch_y.to(device)
            logits = model(batch_x)

            loss = criterion(logits, batch_y)

            val_loss += loss.item() / len(val_loader)
            pred_val.append(logits.softmax(dim=1).cpu().numpy())
            obs_val.append(batch_y.squeeze().cpu().numpy().reshape(-1))
    if scheduler is not None: scheduler.step()
    pred_val = np.concatenate(pred_val,axis=0)
    pred_val_cat=pred_val.argmax(axis=1)
    obs_val = np.concatenate(obs_val)
    auc=roc_auc_score(obs_val,pred_val,multi_class='ovr')
    accuracy=np.mean(obs_val==pred_val_cat)
    return train_loss,val_loss,auc,accuracy

device="cuda" if torch.cuda.is_available() else "cpu"

class Trainer(tune.Trainable):
    def setup(self, config):
        self.model=get_model(config).to(device)
        self.optimizer=get_optimizer(config,self.model)

        self.criterion=nn.CrossEntropyLoss().to(device)
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=50,gamma=0.1)
        self.train_loader,self.val_loader=get_loader(config,train)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def step(self):
        train_loss,loss,auc,accuracy=train_fun(self.model,self.optimizer,self.criterion,
                            self.device,self.train_loader,self.val_loader,self.scheduler)
        return {'loss':loss,'auc':auc,'train_loss':train_loss,'accuracy':accuracy}

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save((self.model.state_dict(),self.optimizer.state_dict()), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        model_state,optimizer_state=torch.load(checkpoint_path)
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)


configs = {
    'dropout':tune.loguniform(0.01,0.5),
    'batch_size':tune.choice([4,8,16,32,]),
    'lr':tune.loguniform(0.00001,0.1),
    'lr_fc':tune.loguniform(0.00001,0.1),
    'l2':tune.loguniform(0.000001,0.5),
    'l2_fc':tune.loguniform(0.000001,0.5),
    'prop_affine':tune.choice([0,0.2,0.5,0.8,1.0]),
    'prop_jitter':tune.choice([0,0.2,0.5,0.8,1.0]),

}
# config={i:v.sample() for i,v in configs.items()}


epochs=150
scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=epochs,
        grace_period=10,
        reduction_factor=2)

reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
result = tune.run(
    Trainer,
    # metric='loss',
    # mode='min',
    checkpoint_at_end=True,
    resources_per_trial={"cpu": 3, "gpu": 0.25},
    config=configs,
    local_dir=os.path.join(log_dir, "Supervised"),
    num_samples=300,
    name=experiment,
    resume=False,
    scheduler=scheduler,
    progress_reporter=reporter,
    reuse_actors=False,
    raise_on_failed_trial=False)


metric="accuracy";mode="max"
best_config=result.get_best_config(metric,mode)

df = result.results_df
# df.to_csv(os.path.join(data_dir, "results/hypersearch.csv"), index=False)
best_trial = result.get_best_trial(metric, mode, "last")
print(best_trial.last_result)

# best_model=get_model(best_config)
best_trainer=Trainer(best_config)

train_dataset=PerchWideDataset(train,label_var='labels',transforms=train_transforms(best_config))
train_loader=DataLoader(train_dataset,shuffle=True,batch_size=best_config['batch_size'],num_workers=15)

best_trainer.train_loader=train_loader
best_trainer.val_loader=test_loader

metrics=[]
for i in range(epochs):
    metrics_=best_trainer.step()
    print(f"Epoch: {i+1} of {epochs} | train loss: {metrics_['train_loss']} | test loss: {metrics_['loss']} | AUC: {metrics_['auc']} | Accuracy: {metrics_['accuracy']}")
    metrics.append(metrics_)


# best_checkpoint=result.get_best_checkpoint(best_trial,metric,mode)
# model_state,optimizer_state=torch.load(best_checkpoint)

best_model=best_trainer.model
best_model.to(device)
# Test model accuracy

best_model.eval()
pred_test=[]
obs_test=[]
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device, dtype=torch.float), batch_y.to(device)
        logits = best_model(batch_x)
        pred_test.append(logits.softmax(dim=1).cpu().numpy())
        obs_test.append(batch_y.squeeze().cpu().numpy().reshape(-1))

pred_test = np.concatenate(pred_test)
obs_test = np.concatenate(obs_test)
pred_test_cat=pred_test.argmax(axis=1)
auc=roc_auc_score(obs_test,pred_test,multi_class='ovr')
aucs=[roc_auc_score(obs_test==i,pred_test[:,i]) for i in range(5)]
accs=[accuracy_score(obs_test==i,pred_test.argmax(axis=1)==i) for i in range(5)]
accuracy=np.mean(obs_test==pred_test_cat)
print(f"Accuracy: {accuracy:.2f}, AUC: {auc:.2f}")
save_table3(experiment=experiment,initialize="Imagenet",model=base_model,
            accuracy=accuracy,auc=auc,config=json.dumps(best_config),
            details=json.dumps({'hostname':os.uname()[1],'aucs':aucs,'accs':accs}))
torch.save(best_model.state_dict(),os.path.join(result_dir,f"weights/{experiment}.pth"))