import os
import pandas as pd
import numpy as np

from settings import perch_image_path,path_var,label_var,data_path

resize_path=os.path.join(data_path,"data/perch_resized")
# PERCH
assessors=pd.read_excel(os.path.join(perch_image_path, "PERCHCXR_RevIDs.xlsx"))
bangladesh = pd.read_csv(os.path.join(perch_image_path, "./Bangladesh/0_Data/PERCH_CXR_BAN.CSV"))
south_africa = pd.read_csv(os.path.join(perch_image_path, "./South Africa/0_Data/PERCH_CXR_SAF.CSV"))
mali = pd.read_csv(os.path.join(perch_image_path, "./Mali/0_Data/PERCH_CXR_MAL.CSV"))
zambia = pd.read_csv(os.path.join(perch_image_path, "./Zambia/0_Data/PERCH_CXR_ZAM.CSV"))
kenya = pd.read_csv(os.path.join(perch_image_path, "./Kenya/0_Data/PERCH_CXR_KEN.CSV"))
thailand = pd.read_csv(os.path.join(perch_image_path, "./Thailand/0_Data/PERCH_CXR_THA.CSV"))
gambia = pd.read_csv(os.path.join(perch_image_path, "./Gambia/0_Data/PERCH_CXR_GAM_CRES.CSV"))

image_files = {}
for root, dirs, files in os.walk(perch_image_path):
    for file in files:
        if file.lower().endswith("jpg"):
            image_files[file.split(".")[0]] = file

perch = pd.concat([bangladesh, south_africa, mali, zambia, kenya, thailand,gambia])
perch = perch[~((perch.DATANOIMG == 1) | (perch.IMGNODATA == 1))]
perch['directory'] = perch.SITE.map(
    {'BAN': "Bangladesh", 'SAF': "South Africa", 'MAL': "Mali",
     'ZAM': 'Zambia', 'KEN': 'Kenya', 'THA': 'Thailand','GAM':'Gambia'})
perch['file_name'] = perch.CXRIMGID.map(image_files)
perch[path_var] = perch[['directory', 'file_name']].aggregate(
    lambda x: np.nan if pd.isna(x[1]) else os.path.join(perch_image_path,x[0], x[1]), axis=1)
perch['resized_path']=perch['file_name'].map(lambda f: np.nan if pd.isna(f) else os.path.join(resize_path,f))
perch[label_var] = perch.FINALCONC.astype("int")-1



assessors2=pd.merge(assessors,perch[['CXRIMGID','PATID','REV1', 'REV2','ARB1', 'ARB2','labels','path','resized_path']],
                    how='right',on="CXRIMGID",suffixes=("_id","_value"))
assessors3a=pd.melt(assessors2,id_vars=['CXRIMGID','PATID','labels','path','resized_path'],
                   value_vars=['REV1_id', 'REV2_id', 'ARB1_id', 'ARB2_id'],
                   var_name='rev_variable',value_name='rev_id')
assessors3a.dropna(axis=0,how='all',subset=['rev_id'],inplace=True)
assessors3a['rev_variable']=assessors3a['rev_variable'].replace('_id$','',regex=True)
assessors3a.rename(columns={'rev_variable':'reviewer'},inplace=True)


assessors3b=pd.melt(assessors2,id_vars=['CXRIMGID','PATID','labels','path','resized_path'],
                   value_vars=['REV1_value', 'REV2_value', 'ARB1_value', 'ARB2_value'],
                   var_name='value_var',value_name='rev_value')
assessors3b.dropna(axis=0,how='all',subset=['rev_value'],inplace=True)
assessors3b['value_var']=assessors3b['value_var'].replace('_value$','',regex=True)
assessors3b.rename(columns={'value_var':'reviewer'},inplace=True)

assessors3=pd.merge(assessors3a,assessors3b,on=['CXRIMGID','PATID','labels','path','reviewer','resized_path'])
assessors3['rev_label']=assessors3['rev_value'].map(lambda x: int(x-1))
assessors3['reviewer']=assessors3['rev_id'].map({v:k for k,v in enumerate(np.sort(assessors3['rev_id'].unique()))})

assessors4=pd.merge(assessors3,perch[['CXRIMGID','SITE']],how='left',on=['CXRIMGID'])
assessors_summ=pd.pivot_table(assessors4,values='CXRIMGID',columns=['SITE'],index=['rev_id'],aggfunc=len,fill_value=0)

def get_site(rev):
    if "ARB" in rev:
        return np.nan
    return np.where(assessors_summ.loc[rev]==0)[0][0] + 1

reviewers=np.unique(assessors4['rev_id'])
rev_sites=list(map(get_site,reviewers))
[f"{r}-SITE{s}" for r,s in zip(reviewers,rev_sites)]

assessors4['rev_site']=assessors4['rev_id'].map(get_site)

accuracy=assessors3.groupby('rev_id').apply(lambda df: np.mean(df['labels']==df['rev_label']))
accuracy.agg({'min':np.min,'max':np.max,'median':np.median})


assessor_summary=assessors3.groupby('rev_id').apply(lambda df: pd.Series({
    'prop':np.mean(df['labels']==df['rev_label']),
    'N':df.shape[0]}))
#summary of number of images assesses
assessor_summary['N'][:14].agg(['min','max'])
assessor_summary['N'][14:].agg(['min','max'])
#summry of accuracy
assessor_summary['prop'][:14].agg(['median','min','max'])
assessor_summary['prop'][14:].agg(['median','min','max'])
#ARBITATOR AGREEMENT
perch.dropna(subset=['REV1','REV2']).apply(lambda df2:df2['REV1']==df2['REV2'],axis=1).mean()
perch.dropna(subset=['ARB1','ARB2']).apply(lambda df2:df2['ARB1']==df2['ARB2'],axis=1).mean()

#image view by site
perch.groupby('SITE')["IMGVIEW"].agg(lambda x: np.mean(x=='AP'))

perch.to_csv(os.path.join(perch_image_path,"perch_wide.csv"),index=False)
assessors4.to_csv(os.path.join(perch_image_path,"perch_long.csv"),index=False)

