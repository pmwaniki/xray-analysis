import os
import shutil

from PIL import Image


from settings import perch_image_path,data_path

source_directory=perch_image_path
target_directory=os.path.join(data_path,"data/perch_resized")
target_directory2=os.path.join(data_path,"data/perch_resized600")
shutil.rmtree(target_directory,ignore_errors=True)
os.makedirs(target_directory, exist_ok=True)
shutil.rmtree(target_directory2,ignore_errors=True)
os.makedirs(target_directory2, exist_ok=True)
size=300
size2=600

image_files = {}
for root, dirs, files in os.walk(source_directory):
    files = [f for f in files if not f[0] == '.']
    for file in files:
        if file.lower().endswith("jpg"):
            image_files[file.split(".")[0]] = file
            image=Image.open(os.path.join(root,file))
            image = image.convert("RGB")
            image300 = image.resize((size, size))
            image600 = image.resize((size2,size2))
            image300.save(os.path.join(target_directory,file))
            image600.save(os.path.join(target_directory2, file))
