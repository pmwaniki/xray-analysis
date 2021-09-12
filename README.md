# Classification of pediatric chest x-rays using machine learning

## Baseline models
ResNet18, ResNet34 and ResNet50  model architectures from the torchvision version 0.8.2 library are used for all our experiments. The ResNet models' last fully connected layer was replaced with a fully connected layer with five output units – one for each WHO category(consolidation; other infiltrate; both consolidation and other infiltrate; normal or uninterpretable).

ASHA algorith in ray[tune] library was to select optimal hyper-parameters for dropout, batch size, learning rate, weight decay and proportion of images with augmentation. The models were trained for 150 epochs, with learning rate halving after 50 and 100 epochs. We trained the models Adam optimizer and cross entropy loss.

## Incorporating individual reader annotation
We extended the ResNet model by adding an embedding layer for reader identifiers. The model could therefore classify a given CXR conditional on reader identifier. Reader identifiers were embedded into a vector of 16 units. The embeddings were then projected to have the same dimensions as the image embeddings using a fully connected layer. Sigmoid activation was applied to the transformed embeddings. Image embeddings were derived by applying global average pooling to the last convolutional layer. Image embeddings and reader embeddings were combined by element-wise multiplication. A fully connected layer with softmax activation was then appended for prediction.
![alt text]()