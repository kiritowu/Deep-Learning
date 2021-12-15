# EfficientNetV2 with Image Classification Benchmarks
### SOTA Model and Training Procedure

Author : Wong Zhao Wu, Bryan

## Abstract
Improving the performance of a vision model is product of the following three different sections [(Cited from Revisiting ResNets)](https://arxiv.org/pdf/2103.07579.pdf).
1. Model Architecture
2. Training Procedure
3. Scaling Method *(Not Enough Computation Power to Explore)*

In CA1, the goal is to implement state of the art model architecture, training procudure and dataset augmentation without compromising the generalisation capability and performance of the model. 

To achieve the goal, I have make use of the [EfficientNetV2 (Tan, M., & Le, Q. V., 2021)](https://arxiv.org/abs/2104.00298) model family that has achieve that has achieved 99% accuracy and comparable to other ViT models from [paperswithcode CIFAR-10 Benchmark Leaderboard](https://paperswithcode.com/sota/image-classification-on-cifar-10). EfficientNetV2 is a Convolutional Network Family that is production of Neural Architecture Search (NAS) with implementations of latest modelling techniques like Fused-MBConv Net from [MobileNetV2 Paper (Sandler. etgi al, 2018)](https://arxiv.org/abs/1801.04381v4) with [Squeeze-and-Excitation (Hu et al.,
2018)](https://arxiv.org/abs/1709.01507).

As for the training procedure, I have gain inspiration from the EfficientNetV2 paper to introduce **Progressive Learning** to scale the input image resolution *([Progressive Resizing](https://www.bookstack.cn/read/th-fastai-book/spilt.3.798d5ac22392691a.md))* and regularization *([RandAugment](https://arxiv.org/abs/1909.13719), [Dropout](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/), [Stochastic Depth](https://arxiv.org/abs/1603.09382), L2 Regularization)* progressively to speed up the training without causing accuracy drop.

## Keywords
- Image Classification
- EfficientNetV2
- RandAugment
- Progressive Learning
- Pytorch
- Timm


## Personal Learning Reflection

Although I did not managed to achieve the same performance as the leaderboard with proper training-validation-test split convention, I managed to achieve accuracy of 92.55% on 10,000 test set and gain practical experience in training and building model in Pytorch.

Written By : Wong Zhao Wu
Last Modified : 15 Dec 2021

![funny-west-highland-white-terrier-dog-decorated-with-photo-props-sits-picture](https://media.istockphoto.com/photos/funny-west-highland-white-terrier-dog-decorated-with-photo-props-sits-picture-id1292884801?b=1&k=20&m=1292884801&s=170667a&w=0&h=L5QgEFpFN1be2Qx8Q9PUWolafU_ecaqYiNwga6eoqxs=)
Image retrieved from [Unsplash](https://www.istockphoto.com/photo/funny-west-highland-white-terrier-dog-decorated-with-photo-props-sits-near-orange-gm1292884801-387530317?utm_source=unsplash&utm_medium=affiliate&utm_campaign=srp_photos_top&utm_content=https%3A%2F%2Funsplash.com%2Fs%2Fphotos%2Fdog&utm_term=dog%3A%3A%3A).