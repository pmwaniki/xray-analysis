import os

data_path="/home/pmwaniki/data/PERCH"
label_var="labels"
path_var="path"
result_dir="/home/pmwaniki/data/PERCH/results"
output_dir='/home/pmwaniki/Dropbox/tmp/xray'


log_dir='/home/pmwaniki/data/logs/perch'


perch_image_path=os.path.join(data_path,"data/perch/")

perch_labels={
        0:'Consolidation',
        1:'Other Infiltrate',
        2:'Consolidation and Other Infiltrate',
        3:'Normal',
        4:'Uninterpretable'}