import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from settings import data_path,perch_image_path,log_dir,result_dir,output_dir
from datasets import PerchLongDataset
from modules import Net2,Net

from sklearn.metrics import roc_auc_score

import ray
# ray.init( num_cpus=12,dashboard_host="0.0.0.0")
ray.init(address="auto")
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler,HyperBandScheduler,AsyncHyperBandScheduler

from utils import save_table3,perch_confusion_matrix
display=os.environ.get("DISPLAY",None)

perch_long=pd.read_csv(os.path.join(perch_image_path,"perch_long.csv"))
perch_wide=pd.read_csv(os.path.join(perch_image_path,"perch_wide.csv"))
rng=np.random.RandomState(500)
all_ids = perch_long['PATID'].unique()
train_ids_ = rng.choice(all_ids, size=int(len(all_ids) * 0.8), replace=False)
train = perch_long[perch_long['PATID'].isin(train_ids_)]
test = perch_long[~perch_long['PATID'].isin(train_ids_)]

input_size=224
base_model="resnet18"
experiment=f"Perch-ensemble-{base_model}b"



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

def get_loader(config,train_data):

    all_ids = train_data['PATID'].unique()
    train_ids_ = rng.choice(all_ids, size=int(len(all_ids) * 0.8), replace=False)
    train_csv = train_data[train_data['PATID'].isin(train_ids_)]
    val_csv = train_data[~train_data['PATID'].isin(train_ids_)]
    train_ds=PerchLongDataset(train_csv,label_var='rev_label',transforms=train_transforms(config),sample_by="CXRIMGID")
    val_ds=PerchLongDataset(val_csv, label_var='rev_label',transforms=val_transforms)



    train_loader = DataLoader(train_ds,
                              batch_size=int(config["batch_size"]),
                              shuffle=True, num_workers=5)
    val_loader = DataLoader(val_ds,
                            batch_size=int(config["batch_size"]),
                            shuffle=False, num_workers=5)
    return train_loader,val_loader

def get_model(config):
    # model=Net2(base_model=base_model,embedding_dim=config['embed_dim'],dropout=config['dropout'])
    model = Net2(base_model=base_model, embedding_dim=config['embed_dim'], dropout=config['dropout'],
                 max_norm=config['max_norm'],activation_fun=config['activation_fun'],
                 comb_embedd='multiply')
    # model = Net(base_model=base_model, dropout=config['dropout'])
    return model

def get_optimizer(config,model):
    optimizer=torch.optim.Adam([
        {'params':model.encoder_model.parameters()},
        # {'params':model.embedding.parameters(),'weight_decay':config['l2_embeddings'],'lr':config['lr_embeddings']},
        {'params': model.embedding.parameters(), 'weight_decay': 0,
         'lr': config['lr_embeddings']},
        {'params':model.fc.parameters(),'weight_decay':config['l2_fc'],'lr':config['lr_fc']},
        {'params':model.linear_embedding.parameters(),'weight_decay':config['l2_embeddings_fc'],'lr':config['lr_embeddings_fc']}
    ], weight_decay=config['l2'],lr=config['lr'])
    return optimizer

def train_fun(model,optimizer,criterion,device,train_loader,val_loader,scheduler=None):
    train_loss = 0

    model.train()
    # print("epoch: %d >> learning rate at beginning of epoch: %.5f" % (epoch, optimizer.param_groups[0]['lr']))
    for batch_x,batch_rev, batch_y in train_loader:
        batch_x,batch_rev,batch_y = batch_x.to(device, dtype=torch.float),batch_rev.to(device),batch_y.to(device)
        logits = model(batch_x,batch_rev)
        # logits = model(batch_x)

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
        for batch_x,batch_rev,batch_y in val_loader:
            batch_x,batch_rev, batch_y = batch_x.to(device, dtype=torch.float), batch_rev.to(device), batch_y.to(device)
            logits = model(batch_x,batch_rev)
            # logits = model(batch_x)

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
        self.optimizer=get_optimizer(config, self.model)

        self.criterion=nn.CrossEntropyLoss().to(device)
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=50,gamma=0.1)
        self.train_loader,self.val_loader=get_loader(config,train)
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def step(self):
        train_loss,loss,auc,accuracy=train_fun(self.model,self.optimizer,self.criterion,
                            device,self.train_loader,self.val_loader,self.scheduler)
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
    'dropout': tune.loguniform(0.01, 0.5),
    'batch_size': tune.choice([4, 8, 16, 32,]),
    'lr': tune.loguniform(0.00001, 0.1),
    'lr_embeddings': tune.loguniform(0.00001, 0.1),
    'lr_fc': tune.loguniform(0.00001, 0.1),
    'lr_embeddings_fc': tune.loguniform(0.00001, 0.1),
    'l2': tune.loguniform(0.000001, 0.5),
    # 'embed_dim': tune.choice([8, 16, 32]),
    'embed_dim': tune.choice([32,]),
    # 'l2_embeddings': tune.loguniform(0.000001, 0.5),
    'max_norm': tune.choice([0.005,0.05,0.5,1,2,4]),
    'l2_embeddings_fc': tune.loguniform(0.000001, 0.5),
    'l2_fc': tune.loguniform(0.000001, 0.5),
    'prop_affine': tune.choice([0, 0.2, 0.5,0.8,1.0]),
    'prop_jitter': tune.choice([0, 0.2, 0.5,0.8,1.0]),
    'activation_fun': tune.choice(['relu',None,'sigmoid','tanh']),
    # 'comb_embedd': tune.choice(['multiply','add'])

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
    resources_per_trial={"cpu": 4, "gpu": .3},
    config=configs,
    local_dir=os.path.join(log_dir, "Supervised"),
    num_samples=300,
    name=experiment,
    resume=True,
    scheduler=scheduler,
    progress_reporter=reporter,
    # reuse_actors=False,
    raise_on_failed_trial=False)

metric="accuracy";mode="max"
best_config=result.get_best_config(metric,mode)

df = result.results_df
# df.to_csv(os.path.join(data_dir, "results/hypersearch.csv"), index=False)
best_trial = result.get_best_trial(metric, mode, "last")
print(best_trial.last_result)

# best_model=get_model(best_config)
best_trainer=Trainer(best_config)

train_dataset=PerchLongDataset(train,label_var='rev_label',transforms=train_transforms(best_config),sample_by="CXRIMGID")
train_loader=DataLoader(train_dataset,shuffle=True,batch_size=best_config['batch_size'],num_workers=15)
val_dataset=PerchLongDataset(test,label_var='rev_label',transforms=val_transforms,sample_by=None)
val_loader=DataLoader(val_dataset,shuffle=False,batch_size=best_config['batch_size'],num_workers=15)

best_trainer.train_loader=train_loader
best_trainer.val_loader=val_loader

metrics=[]
for i in range(epochs):
    metrics_=best_trainer.step()
    print(f"Epoch: {i+1} of {epochs} | train loss: {metrics_['train_loss']} | test loss: {metrics_['loss']} | AUC: {metrics_['auc']} | Accuracy: {metrics_['accuracy']}")
    metrics.append(metrics_)

test_expanded=test[['CXRIMGID','labels','resized_path','SITE']].drop_duplicates(inplace=False)
test_expanded['reviewer']=np.nan
test_expanded2=[]

for i in range(18):
    temp=test_expanded.copy()
    temp['reviewer']=i
    test_expanded2.append(temp)

test_expanded2=pd.concat(test_expanded2,axis=0,ignore_index=True)


test_dataset=PerchLongDataset(test_expanded2,label_var=None,transforms=val_transforms,sample_by=None)
test_loader=DataLoader(test_dataset,shuffle=False,batch_size=best_config['batch_size'],num_workers=15)

# best_checkpoint=result.get_best_checkpoint(best_trial,metric,mode)
# model_state,optimizer_state=torch.load(best_checkpoint)

best_model=best_trainer.model
# best_model.load_state_dict(torch.load(os.path.join(result_dir,f"weights/{experiment}.pth")))
best_model.to(device)
# Test model accuracy

best_model.eval()
pred_test=[]
obs_test=[]
with torch.no_grad():
    for batch_x, batch_rev in test_loader:
        batch_x, batch_rev = batch_x.to(device, dtype=torch.float), batch_rev.to(device)
        logits = best_model(batch_x, batch_rev)
        pred_test.append(logits.softmax(dim=1).cpu().numpy())

pred_test = np.concatenate(pred_test)
for i in range(5):
    test_expanded2[f'pred_class_{i}']=pred_test[:,i]

test_pred_agg=test_expanded2.groupby(['CXRIMGID','SITE'])['labels', 'pred_class_0','pred_class_1', 'pred_class_2', 'pred_class_3', 'pred_class_4'].agg(np.mean)
test_pred_agg=test_pred_agg.reset_index()
test_pred_agg=pd.merge(test_pred_agg,perch_wide[['CXRIMGID',"_AGEM"]],how="left",on="CXRIMGID")
test_pred_agg['age12m']=test_pred_agg['_AGEM']<12
test_pred_agg['pred_cat']=np.argmax(test_pred_agg[['pred_class_0','pred_class_1', 'pred_class_2', 'pred_class_3', 'pred_class_4']].values,axis=1)
test_pred_agg['correct']=test_pred_agg['labels']==test_pred_agg['pred_cat']


auc=roc_auc_score(test_pred_agg['labels'].values,
                  test_pred_agg[['pred_class_0','pred_class_1', 'pred_class_2', 'pred_class_3', 'pred_class_4']].values
                  ,multi_class='ovr')
aucs=[roc_auc_score(test_pred_agg['labels'].values==i,test_pred_agg[f'pred_class_{i}'].values) for i in range(5)]
accuracy=test_pred_agg['correct'].mean()
site_accuracies=test_pred_agg.groupby('SITE')[['correct']].mean()
age_accuracies=test_pred_agg.groupby('age12m')[['correct']].mean()
print(f"Accuracy: {accuracy:.2f}, AUC: {auc:.2f}")
save_table3(experiment=experiment,initialize="Imagenet",model=base_model,
            accuracy=accuracy,auc=auc,config=json.dumps(best_config),
            details=json.dumps({'hostname':os.uname()[1],'aucs':aucs}))
torch.save(best_model.state_dict(),os.path.join(result_dir,f"weights/{experiment}.pth"))

embeddings=best_model.embedding.weight.detach().cpu().numpy()
labelList = ['REV1-SITE3', 'REV2-SITE2', 'REV3-SITE4', 'REV4-SITE7', 'REV5-SITE5', 'REV6-SITE6', 'REV7-SITE1',
             'REV8-SITE3', 'REV9-SITE2', 'REV10-SITE4', 'REV11-SITE7', 'REV12-SITE5', 'REV13-SITE6', 'REV14-SITE1',
             'ARB1', 'ARB2', 'ARB3', 'ARB4']

corr=np.corrcoef(embeddings)

fig,ax=plt.subplots(1,1,figsize=(10,10))
s=ax.imshow(corr,vmin=-1.0,vmax=1.0,cmap=plt.get_cmap("seismic"))
plt.xticks(range(18),labels=labelList,rotation=90)
plt.yticks(range(18),labels=labelList)
fig.colorbar(s)
plt.savefig(os.path.join(output_dir,f"correlation plot - {experiment}.png"))
if display:
    plt.show()
else:
    plt.close()

if display:
    fig=perch_confusion_matrix(test_pred_agg['labels'].values,test_pred_agg['pred_cat'].values)
    fig.savefig(os.path.join(output_dir,f"{experiment}-confussion matrix.png"))

if display:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    y = lowess(test_pred_agg['correct'] * 1.0, test_pred_agg['_AGEM'], )
    plt.plot(y[:, 0], y[:, 1])
    plt.xlabel("Age in months")
    plt.ylabel("Classification accuracy")
    plt.savefig(os.path.join(output_dir,f"{experiment}- model accuracy vs age.png"))
    plt.show()