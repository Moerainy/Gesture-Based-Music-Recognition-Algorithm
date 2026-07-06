![1](imgs/slide0.jpg)
可以看到，我们尝试了3种模型来解决这个分类问题，
每一种模型代表了一种对手势动作时空信息的理解方式。

We tried three different models to solve this classification problem. Each model represents a different way to understand the spatial and temporal information in gesture actions.


![2](imgs/slide1.jpg)
第一个模型，也是我们的baseline模型，是logistics regression。
它是一个简单的线性模型，通过将视频像素整体展平成一个向量，
然后通过线性映射加softmax实现分类。
这个模型并没有尝试主动理解动作的时空结构。


The first model, which is also our baseline model, is logistic regression. It flattens all pixels into one vector, and then uses a linear mapping and softmax for classification. This model does not actively learn the spatial or temporal structure of the actions.



![3](imgs/slide2.jpg)
第二个模型是卷积神经网络和LSTM模型的结合，
这个模型通过两个独立的模型，分别提取动作的空间和时间特征。
它通过一个ResNet-18模型，将每一帧画面嵌入到一个512维的向量，
来实现提取图像的空间特征的目的。
然后再将这些向量放入LSTM网络中提取时间特征。
最后用一个线性层加softmax实现分类。

The second model combines a CNN with an LSTM model.  This model uses two separate networks to extract spatial and temporal features. First, a ResNet-18 model embeds each video frame into a 512-dimensional vector to extract spatial features from the frames.  Then these vectors are passed into an LSTM network to learn temporal features. Finally, the distribution is estimated by a fully connectedn layer.


![4](imgs/slide3.jpg)
下面我们来看一下这个网络的整体结构。
首先看ResNet-18，图像数据输入进网络后，
会首先经过一个7x7的全局卷积和3x3的MaxPool。
然后数据依经过4个stages共16次卷积，
其中每两次卷积操作会进行一次残差连接，是为一个basic block，
其中残差会经过一次shortcut。
每一stage包含两个block，数据经过一个stage后通道数加倍，分辨率减半。
最后再对每一个通道做一次平均池化，然后我们就得到了每一帧图像的特征向量。
最后再将这些特征向量放入LSTM中提取时间特征，
然后将最后一个中间状态，也就是$h_T$，放入线性层得到终的分类结果。

Now let us look at the overall structure of this network.  First is the ResNet-18 part.  The image data first passes through a 7×7 convolution layer and a 3×3 max pooling layer.  Then the data goes through 4 stages with a total of 16 convolution operations.  Every two convolution layers form a basic block with a residual connection through a shortcut path.  Each stage contains two blocks.  After each stage, the number of channels doubles while the resolution is reduced by half.  At the end, average pooling is applied to each channel, and we obtain the feature vector for each frame.  These feature vectors are then fed into the LSTM to extract temporal features.  Finally, the last hidden state, which is \(h_T\), is passed into a linear layer to get the final classification result.

![5](imgs/slide4.jpg)
下面是我们的第三个模型，这是一个spatiotemporal卷积神经网络。
它的架构和ResNet-18非常相似，特别是stages和block结构，以及残差连接。
唯一的不同是basic block的构成。


The third model is a spatiotemporal convolutional neural network. This model has a  Its structure is very similar to ResNet-18, especially in the stage structure and residual connections.  The main difference is the design of the basic block.

![6](imgs/slide5.jpg)
不同于一般的full 3d CNN模型，
R(2+1)D CNN的basic block采取了先空间后时间的卷积方法，
使用远小于full 3D CNN的计算量达成了相近的效果。
为了加速训练的过程，我们使用了在Kinestic-400数据集上预训练过的模型，
仅仅将模型最后的fc layer替换为符合我们的数据集维度的大小。


Unlike a standard full 3D CNN model,  the R(2+1)D CNN basic block uses spatial convolution first and temporal convolution second.  This design achieves performance similar to a full 3D CNN while using much less computation.  To speed up training, we used a model pretrained on the Kinetics-400 dataset.  We only replaced the final fully connected layer so that it matches the number of classes in our dataset.


![7](imgs/slide6.jpg)
为了处理各类别样本数量不平衡的问题，我们使用了加权交叉熵函数作为损失函数。
其中各类别权重与该类别样本数量成反比，并乘以N/K的常数scalar，
以控制权重绝对值处于合理范围内。
其中N为训练集样本总数，K为类别数量。

To deal with the imbalance between classes, we employed a weighted cross-entropy as the loss function.  The weight for each class is inversely proportional to the number of samples in that class.  We also multiplied the weights by a scalar value \(N/K\) to keep the weights in a reasonable range.  Here, \(N\) is the total number of training samples, and \(K\) is the number of classes.



![8](imgs/slide7.jpg)
下面是一些超参数的设定，
由于R(2+1)D模型已经经过预训练，所以我们为其赋予了一个较小的初始学习率。
Regularization方面，我们为logistics regression设置了较大的L2 penalty，
以抑制其过拟合的趋势。
Batch Size的设定的主要考量因素是内存占用，因为我只有8GB VRAM。


These are some of our hyperparameter settings. Since the R(2+1)D model was already pretrained, we used a smaller initial learning rate for it.  For regularization, we applied a larger L2 penalty to logistic regression to reduce overfitting.  The main factor when choosing the batch size was memory usage, because I only had 8GB of VRAM.