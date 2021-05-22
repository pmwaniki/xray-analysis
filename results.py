import os
import json
import sqlite3
import pandas as pd
import numpy as np
from settings import output_dir,perch_labels


db=sqlite3.connect("results.db")
cur=db.cursor()
cur.execute("SELECT * FROM table3")

rows=cur.fetchall()

table=pd.DataFrame(rows,columns=['experiment','init','model','acc','auc','config','details'])
table['ensemble']=table['experiment'].map(lambda x: "Yes" if "ensemble" in x else "No")
table=table.loc[table['experiment'].isin(['Perch-resnet18', 'Perch-resnet50','Perch-resnet34',
       'Perch-ensemble-resnet18b', 'Perch-ensemble-resnet34b',
       'Perch-ensemble-resnet50b',]),:]
table['model']=table['model'].map(lambda x:x.replace("resnet","ResNet"))


table1=pd.melt(table,id_vars=['model','ensemble',],value_vars=['acc','auc'])
table1['value']=table1['value'].round(2)

table1b=pd.pivot_table(table1,index=['model'],columns=['variable','ensemble'],fill_value=np.nan)
table1b.to_csv(os.path.join(output_dir,"Table1.csv"))

table2=[]
for i,row in table.iterrows():
    aucs=json.loads(row['details'])['aucs']
    r={perch_labels[i]:v for i,v in enumerate(aucs)}
    r['model']=row['model']
    r['ensemble']=row['ensemble']
    table2.append(r)

table2=pd.DataFrame(table2)
table2b=pd.melt(table2,id_vars=['model','ensemble'],value_vars=perch_labels.values(),value_name="AUC")
table2b['AUC']=table2b['AUC'].round(2)
table2c=pd.pivot_table(table2b,values='AUC',index=['model','variable'],columns=['ensemble'])

table2c.to_csv(os.path.join(output_dir,'Table2.csv'),index=True)

table1c=table1.copy()
table1c=table1c.drop(index=np.where(table1c['variable']=='auc')[0])
table1c=table1c.rename(columns={'value':"Accuracy"})
table3=pd.merge(table2b,table1c,how='left',on=['model','ensemble'])
table3b=pd.melt(table3,id_vars=['model','ensemble','variable_x'],value_vars=['Accuracy',"AUC"])
table3c=pd.pivot_table(table3b,index=['model','variable_x'],columns=['variable','ensemble'],values='value')
table3c.to_csv(os.path.join(output_dir,'Table3.csv'),index=True)
# BEST MODEL
print(table.loc[table['acc']==np.max(table['acc']),"experiment"])

table4=[]
for i,row in table.iterrows():
    r=json.loads(row['config'])
    # r={perch_labels[i]:v for i,v in enumerate(aucs)}
    r['model']=row['model']
    r['ensemble']=row['ensemble']
    table4.append(r)

table4=pd.DataFrame(table4)
table4=table4.drop(columns=['embed_dim'])
table4['dropout']=table4['dropout'].map(lambda x:f"{x:.2f}")
table4['lr']=table4['lr'].map(lambda x:f"{x:.6f}")
table4['lr_fc']=table4['lr_fc'].map(lambda x:f"{x:.6f}")
table4['lr_embeddings_fc']=table4['lr_embeddings_fc'].map(lambda x:f"{x:.6f}" if not np.isnan(x) else '')
table4['lr_embeddings']=table4['lr_embeddings'].map(lambda x:f"{x:.6f}" if not np.isnan(x) else '')
table4['prop_affine']=table4['prop_affine'].map(lambda x:f"{x:.1f}")
table4['prop_jitter']=table4['prop_jitter'].map(lambda x:f"{x:.1f}")

table4['l2']=table4['l2'].map(lambda x:f"{x:.6f}")
table4['l2_fc']=table4['l2_fc'].map(lambda x:f"{x:.6f}")
table4['l2_embeddings_fc']=table4['l2_embeddings_fc'].map(lambda x:f"{x:.6f}" if not np.isnan(x) else '')
table4['batch_size']=table4['batch_size'].map(lambda x:f"{x}" if not np.isnan(x) else '')
table4['max_norm']=table4['max_norm'].map(lambda x:f"{x}" if not np.isnan(x) else '')
table4.loc[(table4['ensemble']=="Yes") & (table4['activation_fun'].isna()),'activation_fun']="identity"
table4['activation_fun']=table4['activation_fun'].fillna('')



table4b=pd.melt(table4,id_vars=['model','ensemble'])
table4b['param'] = table4b['variable'].map({
    'batch_size': 'Batch size',
    'dropout': "Dropout",
    'embed_dim': "Embedding dimension",
    'l2': 'L2 regularization of convolutional layers',
    'l2_embeddings_fc': "L2 regularization of fully connected layer projecting the reader embeddings",
    'l2_fc': "L2 regularization of fully connected layer",
    'lr': "Learning rate for convolutional layers",
    'lr_embeddings': "Learning rate for reader embeddings",
    'lr_embeddings_fc': "Learning rate for fully connected layer projecting the reader embeddings",
    'lr_fc': "Learning rate for fully connected layer", 'max_norm': "Max L2-norm of reader Embeddings",
    'prop_affine': "Proportion of training images with affine transformation augmentation",
    'prop_jitter': "Proportion of images with color brightness and contrast augmentation",
'activation_fun':"Activation function for projected reader embeddings"}
)
table4b['value2']=table4b['value'].fillna("").values
table4c=pd.pivot_table(table4b,index=['param'],columns=['model','ensemble'],values='value2',aggfunc=lambda x: ' '.join(x))
table4c.to_csv(os.path.join(output_dir,'hyper-parameters.csv'),index=True)