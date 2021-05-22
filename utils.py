import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.metrics import confusion_matrix


def show_grid(dataset, nrow=2, ncol=3, mean=(0.485, 0.456, 0.406), sd=(0.229, 0.224, 0.225)):
    images = []
    for i in range(nrow * ncol):
        im, _ = dataset.__getitem__(i)
        images.append(im)
    img = torch.stack(images)
    grid = make_grid(img, nrow=nrow, normalize=False)

    npimg = grid.numpy()
    npimg = npimg * np.array(sd)[:, None, None]
    npimg = npimg + np.array(mean)[:, None, None]
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()

def plot_embeddings(embeddings,metric='cosine'):
    linked = linkage(embeddings, 'complete', metric=metric)

    labelList = ['01REV-SITE3', '02REV-SITE2', '03REV-SITE4', '04REV-SITE7', '05REV-SITE5', '06REV-SITE6', '07REV-SITE1', '08REV-SITE3', '09REV-SITE2', '10REV-SITE4', '11REV-SITE7', '12REV-SITE5', '13REV-SITE6', '14REV-SITE1', '15ARB', '16ARB', '17ARB', '18ARB']
    plt.figure(figsize=(10, 7))
    dendrogram(linked,
                orientation='right',
                labels=labelList,
                distance_sort='descending',
                show_leaf_counts=True)
    plt.show()

def perch_confusion_matrix(ytrue,ypred,labels=['Consolidation',
                                                'Other Infiltrate',
                                                'Consolidation \n and \nOther Infiltrate',
                                                'Normal',
                                                'Uninterpretable']):
    cf_matrix=confusion_matrix(ytrue,ypred,labels=[0,1,2,3,4])
    cf_matrix_norm=cf_matrix/np.reshape(cf_matrix.sum(axis=1),[-1,1])

    fig,ax=plt.subplots(1,1,figsize=(12,12))
    img=ax.matshow(cf_matrix_norm,cmap=plt.cm.get_cmap("Greys"),vmin=0,vmax=1)
    for i in [0,1,2,3,4]:
        for j in [0,1,2,3,4]:
            ax.text(j,i,"%d(%.1f)" % (cf_matrix[i,j],cf_matrix_norm[i,j]),ha='center',fontsize=9)
    ax.set_xticks([0,1,2,3,4])
    ax.set_xticklabels(labels,rotation=45,ha="left",rotation_mode="anchor")
    # ax.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #      rotation_mode="anchor")
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(labels)
    ax.set_ylim(4.5,-0.5)
    ax.set_xlabel("Predicted Class",fontsize=14)
    ax.set_ylabel("Target Class",fontsize=14)
    fig.colorbar(img)
    plt.show()

    return fig


def create_table3():
    conn=sqlite3.connect("results.db")
    c=conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS table3 (
        experiment TEXT NOT NULL,
        initialize TEXT NOT NULL,
        model TEXT NOT NULL,
        accuracy NUMERIC,
        auc NUMERIC,
        config TEXT,
        details TEXT NOT NULL ,
        PRIMARY KEY (experiment, initialize)
    )
    """)
    conn.commit()
    conn.close()


def save_table3(experiment, initialize, model, accuracy, auc, config, details=None):
    create_table3()
    conn = sqlite3.connect("results.db")
    c = conn.cursor()
    c.execute("""
                    INSERT OR REPLACE INTO table3 (
                        experiment,
                        initialize,
                        model,
                        accuracy,
                        auc,
                        config,
                        details
                    ) VALUES(?,?,?,?,?,?,?)
                    """, (experiment, initialize, model, accuracy, auc, config, details))
    conn.commit()
    conn.close()
    return None
