import matplotlib.pyplot as plt
import os,shutil
import sqlite3
import json

import pandas as pd
from ray.tune.analysis.experiment_analysis import Analysis
from settings import log_dir,output_dir

display=os.environ.get("DISPLAY",None)

db=sqlite3.connect("results.db")
cur=db.cursor()
cur.execute("SELECT * FROM table3")

rows=cur.fetchall()

table3=pd.DataFrame(rows,columns=['experiment','init','model','acc','auc','config','details'])
table3['ensemble']=table3['experiment'].map(lambda x: "ensemble" in x)
table3=table3.loc[table3['experiment'].isin(['Perch-resnet18', 'Perch-resnet50','Perch-resnet34',
       'Perch-ensemble-resnet18b', 'Perch-ensemble-resnet34b',
       'Perch-ensemble-resnet50b',]),:]


experiments,hosts=table3['experiment'],table3['details'].map(lambda x:json.loads(x)['hostname'])
available_experiments=[e for e,h in zip(experiments,hosts) if h==os.uname()[1]]

log_dirs=os.listdir(os.path.join(log_dir, "Supervised"))
temp_dir='/home/pmwaniki/Dropbox/tmp'
for exp in available_experiments:
    analysis=Analysis(os.path.join(log_dir, "Supervised",exp))
    dfs=analysis.trial_dataframes
    incomplete=[k for k,v in dfs.items() if "loss" not in v]
    for i in incomplete:shutil.rmtree(i,ignore_errors=True) #delete trials without data
    analysis = Analysis(os.path.join(log_dir, "Supervised", exp))
    score = "accuracy"
    mode = "max"
    best_trial = analysis.get_best_logdir(score, mode)
    best_dir = analysis.get_best_logdir(score, mode)
    best_data = analysis.trial_dataframes[best_trial]
    best_config = analysis.get_best_config(score, mode)
    best_data.to_csv(os.path.join(temp_dir,f"best-trial-{exp}.csv"),index=False)



all_experiments={}
for exp in experiments:
    exp_file=os.path.join(temp_dir,f"best-trial-{exp}.csv")
    try:
        exp_data=pd.read_csv(exp_file)
    except FileNotFoundError:
        exp_data=None
    all_experiments[exp]=exp_data

print("Experiments without data: ", [e for e in  experiments if all_experiments[e] is None])

def name_fun(exp):
    ens="ens" in exp
    if "resnet50" in exp:
        name= "ResNet50"
    if "resnet34" in exp:
        name=  "ResNet34"
    if "resnet18" in exp:
        name=  "ResNet18"
    if ens:
        name= name + "- Ensemble"
    return name

def color_fun(exp):
    if "resnet50" in exp:
        return "blue"
    if "resnet34" in exp:
        return "black"
    if "resnet18" in exp:
        return "green"

fig,ax=plt.subplots(1,2,figsize=(12,10))

for exp,dat in all_experiments.items():
    if dat is None: continue
    ax[0].plot(dat['training_iteration'],dat['accuracy'],
               c=color_fun(exp),
               # c="black" if "ens" in exp else "blue",
               label=name_fun(exp),
               linestyle="-" if "ens" in exp else ":",
               linewidth=1)
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epoch")

    ax[1].plot(dat['training_iteration'], dat['loss'],
               c=color_fun(exp),
               label=name_fun(exp),
               linestyle="-" if "ens" in exp else ":",
               linewidth=1)
    ax[1].set_title("loss")
    ax[1].set_xlabel("Epoch")

handles, labels = ax[0].get_legend_handles_labels()
unique_labels=[labels.index(x) for x in set(labels)]
plt.legend([handles[i] for i in unique_labels],[labels[i] for i in unique_labels],bbox_to_anchor=(0.5, -0.05),
           ncol=3, fancybox=True, shadow=True,loc="upper right")
plt.legend(handles,labels,bbox_to_anchor=(0.5, -0.05),
           ncol=3, fancybox=True, shadow=True,loc="upper right")
# fig.legend()
plt.savefig(os.path.join(output_dir,"loss and accuracy trend.png"))
if display: fig.show()
