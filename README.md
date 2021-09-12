# Classification of pediatric chest x-rays using machine learning
The models are trained to classify the Pneumonia Etiology Research for Child Health (PERCH) data-set that contains CXR images of paediatric patients hospitalized with pneumonia (https://clinepidb.org/ce/app/record/dataset/DS_1595200bb8).

## Baseline models
ResNet18, ResNet34 and ResNet50  model architectures from the torchvision version 0.8.2 library are used. The ResNet models' last fully connected layer was replaced with a fully connected layer with five output units – one for each WHO category(consolidation; other infiltrate; both consolidation and other infiltrate; normal or uninterpretable). Analysis code in *perch.py*.

ASHA algorith in ray[tune] library was to select optimal hyper-parameters for dropout, batch size, learning rate, weight decay and proportion of images with augmentation. The models were trained for 150 epochs, with learning rate halving after 50 and 100 epochs. We trained the models Adam optimizer and cross entropy loss.

## Incorporating individual reader annotation
ResNet models were extended by adding an embedding layer for reader identifiers. Reader identifiers were embedded into a vector of 16 units. The embeddings were then projected to have the same dimensions as the image embeddings using a fully connected layer. Image embeddings were derived by applying global average pooling to the last convolutional layer. Image embeddings and reader embeddings were combined by element-wise multiplication. A fully connected layer with softmax activation was then appended for prediction. Analysis code in *perch_ensemble.py*.


![alt text](https://github.com/pmwaniki/xray-analysis/blob/master/perch_ensemble.png)

The model could make multiple prediction for each CXR (one for each reader).There were 18 readers in total. Thus, 18 predictions could be made for every CXR image. During inference, the 18 predictions were then aggregated to give the final prediction using an unweighted mean. 