# Cross Entropy损失函数

这篇文章中，讨论的Cross Entropy损失函数常用于分类问题中，但是为什么它会在分类问题中这么有效呢？我们先从一个简单的分类例子来入手。

## 1. 图像分类任务

我们希望根据图片动物的轮廓、颜色等特征，来预测动物的类别，有三种可预测类别：猫、狗、猪。假设我们当前有两个模型（参数不同），这两个模型都是通过sigmoid/softmax的方式得到对于每个预测结果的概率值：

**模型1**：

| 预测        | 真实       | 是否正确 |
| ----------- | ---------- | -------- |
| 0.3 0.3 0.4 | 0 0 1 (猪) | 正确     |
| 0.3 0.4 0.3 | 0 1 0 (狗) | 正确     |
| 0.1 0.2 0.7 | 1 0 0 (猫) | 错误     |

**模型1**对于样本1和样本2以非常微弱的优势判断正确，对于样本3的判断则彻底错误。

**模型2**：

| 预测        | 真实       | 是否正确 |
| ----------- | ---------- | -------- |
| 0.1 0.2 0.7 | 0 0 1 (猪) | 正确     |
| 0.1 0.7 0.2 | 0 1 0 (狗) | 正确     |
| 0.3 0.4 0.3 | 1 0 0 (猫) | 错误     |

**模型2**对于样本1和样本2判断非常准确，对于样本3判断错误，但是相对来说没有错得太离谱。

好了，有了模型之后，我们需要通过定义损失函数来判断模型在样本上的表现了，那么我们可以定义哪些损失函数呢？

## 1.1 Classification Error（分类错误率）

最为直接的损失函数定义为： ![[公式]](https://www.zhihu.com/equation?tex=classification%5C+error%3D%5Cfrac%7Bcount%5C+of%5C+error%5C+items%7D%7Bcount%5C+of+%5C+all%5C+items%7D)

**模型1：** ![[公式]](https://www.zhihu.com/equation?tex=classification%5C+error%3D%5Cfrac%7B1%7D%7B3%7D)

**模型2：** ![[公式]](https://www.zhihu.com/equation?tex=classification%5C+error%3D%5Cfrac%7B1%7D%7B3%7D)

我们知道，**模型1**和**模型2**虽然都是预测错了1个，但是相对来说**模型2**表现得更好，损失函数值照理来说应该更小，但是，很遗憾的是， ![[公式]](https://www.zhihu.com/equation?tex=classification%5C+error) 并不能判断出来，所以这种损失函数虽然好理解，但表现不太好。

## 1.2 Mean Squared Error (均方误差)

均方误差损失也是一种比较常见的损失函数，其定义为： ![[公式]](https://www.zhihu.com/equation?tex=MSE%3D%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi%7D%5En%28%5Chat%7By_i%7D-y_i%29%5E2)

**模型1：**

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D++++%5Ctext%7Bsample+1+loss%3D%7D%280.3-0%29%5E2+%2B+%280.3-0%29%5E2+%2B+%280.4-1%29%5E2+%3D+0.54+%5C%5C++++%5Ctext%7Bsample+2+loss%3D%7D%280.3-0%29%5E2+%2B+%280.4-1%29%5E2+%2B+%280.3-0%29%5E2+%3D+0.54+%5C%5C++++%5Ctext%7Bsample+3+loss%3D%7D%280.1-1%29%5E2+%2B+%280.2-0%29%5E2+%2B+%280.7-0%29%5E2+%3D+1.34+%5C%5C+%5Cend%7Baligned%7D+%5C%5C)

对所有样本的loss求平均：

![[公式]](https://www.zhihu.com/equation?tex=MSE%3D%5Cfrac%7B0.54%2B0.54%2B1.34%7D%7B3%7D%3D0.81+%5C%5C)

**模型2：**

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+++%26+%5Ctext%7Bsample+1+loss%3D%7D%280.1-0%29%5E2+%2B+%280.2-0%29%5E2+%2B+%280.7-1%29%5E2+%3D+0.14%5C%5C++++%26%5Ctext%7Bsample+2+loss%3D%7D%280.1-0%29%5E2+%2B+%280.7-1%29%5E2+%2B+%280.2-0%29%5E2+%3D+0.14%5C%5C++++%26%5Ctext%7Bsample+3+loss%3D%7D%280.3-1%29%5E2+%2B+%280.4-0%29%5E2+%2B+%280.3-0%29%5E2+%3D+0.74%5C%5C+%5Cend%7Baligned%7D+%5C%5C)

对所有样本的loss求平均：

![[公式]](https://www.zhihu.com/equation?tex=MSE%3D%5Cfrac%7B0.14%2B0.14%2B0.74%7D%7B3%7D%3D0.34+%5C%5C)

我们发现，MSE能够判断出来**模型2**优于**模型1**，那为什么不采样这种损失函数呢？主要原因是在分类问题中，使用sigmoid/softmx得到概率，配合MSE损失函数时，采用梯度下降法进行学习时，会出现模型一开始训练时，学习速率非常慢的情况（https://zhuanlan.zhihu.com/p/35707643）。

有了上面的直观分析，我们可以清楚的看到，对于分类问题的损失函数来说，分类错误率和均方误差损失都不是很好的损失函数，下面我们来看一下交叉熵损失函数的表现情况。

## 1.3 Cross Entropy Loss Function（交叉熵损失函数）

## 1.3.1 表达式

## (1) 二分类

在二分的情况下，模型最后需要预测的结果只有两种情况，对于每个类别我们的预测得到的概率为 ![[公式]](https://www.zhihu.com/equation?tex=p) 和 ![[公式]](https://www.zhihu.com/equation?tex=1-p) ，此时表达式为：

![[公式]](https://www.zhihu.com/equation?tex=L+%3D+%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%7D+L_i+%3D+%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%7D-%5By_i%5Ccdot+log%28p_i%29+%2B+%281-y_i%29%5Ccdot+log%281-p_i%29%5D+%5C%5C)

其中：
- ![[公式]](https://www.zhihu.com/equation?tex=y_i) —— 表示样本 ![[公式]](https://www.zhihu.com/equation?tex=i) 的label，正类为 ![[公式]](https://www.zhihu.com/equation?tex=1) ，负类为 ![[公式]](https://www.zhihu.com/equation?tex=0)
- ![[公式]](https://www.zhihu.com/equation?tex=p_i) —— 表示样本 ![[公式]](https://www.zhihu.com/equation?tex=i) 预测为正类的概率

## (2) 多分类

多分类的情况实际上就是对二分类的扩展：

![[公式]](https://www.zhihu.com/equation?tex=L+%3D+%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%7D+L_i+%3D+-+%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%7D+%5Csum_%7Bc%3D1%7D%5EMy_%7Bic%7D%5Clog%28p_%7Bic%7D%29+%5C%5C)

其中：
- ![[公式]](https://www.zhihu.com/equation?tex=M) ——类别的数量
- ![[公式]](https://www.zhihu.com/equation?tex=y_%7Bic%7D) ——符号函数（ ![[公式]](https://www.zhihu.com/equation?tex=0) 或 ![[公式]](https://www.zhihu.com/equation?tex=1) ），如果样本 ![[公式]](https://www.zhihu.com/equation?tex=i) 的真实类别等于 ![[公式]](https://www.zhihu.com/equation?tex=c) 取 ![[公式]](https://www.zhihu.com/equation?tex=1) ，否则取 ![[公式]](https://www.zhihu.com/equation?tex=0)
- ![[公式]](https://www.zhihu.com/equation?tex=p_%7Bic%7D) ——观测样本 ![[公式]](https://www.zhihu.com/equation?tex=i) 属于类别 ![[公式]](https://www.zhihu.com/equation?tex=c) 的预测概率

现在我们利用这个表达式计算上面例子中的损失函数值：

**模型1**：
![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D++++%5Ctext%7Bsample+1+loss%7D+%3D+-+%280%5Ctimes+log0.3+%2B+0%5Ctimes+log0.3+%2B+1%5Ctimes+log0.4%29+%3D+0.91+%5C%5C++++%5Ctext%7Bsample+2+loss%7D+%3D+-+%280%5Ctimes+log0.3+%2B+1%5Ctimes+log0.4+%2B+0%5Ctimes+log0.3%29+%3D+0.91+%5C%5C++++%5Ctext%7Bsample+3+loss%7D+%3D+-+%281%5Ctimes+log0.1+%2B+0%5Ctimes+log0.2+%2B+0%5Ctimes+log0.7%29+%3D+2.30+%5C%5C+%5Cend%7Baligned%7D+%5C%5C)

对所有样本的loss求平均：

![[公式]](https://www.zhihu.com/equation?tex=L%3D%5Cfrac%7B0.91%2B0.91%2B2.3%7D%7B3%7D%3D1.37+%5C%5C)

**模型2：**

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D++++%5Ctext%7Bsample+1+loss%7D+%3D+-+%280%5Ctimes+log0.1+%2B+0%5Ctimes+log0.2+%2B+1%5Ctimes+log0.7%29+%3D+0.35+%5C%5C++++%5Ctext%7Bsample+2+loss%7D+%3D+-+%280%5Ctimes+log0.1+%2B+1%5Ctimes+log0.7+%2B+0%5Ctimes+log0.2%29+%3D+0.35+%5C%5C++++%5Ctext%7Bsample+3+loss%7D+%3D+-+%281%5Ctimes+log0.3+%2B+0%5Ctimes+log0.4+%2B+0%5Ctimes+log0.4%29+%3D+1.20+%5C%5C+%5Cend%7Baligned%7D+%5C%5C)

对所有样本的loss求平均：

![[公式]](https://www.zhihu.com/equation?tex=L%3D%5Cfrac%7B0.35%2B0.35%2B1.2%7D%7B3%7D%3D0.63+%5C%5C)

可以发现，交叉熵损失函数可以捕捉到**模型1**和**模型2**预测效果的差异。

## 2. 函数性质

![](https://pic3.zhimg.com/80/v2-f049a57b5bb2fcaa7b70f5d182ab64a2_1440w.jpg)

可以看出，该函数是凸函数，求导时能够得到全局最优值。

## 3. 学习过程

交叉熵损失函数经常用于分类问题中，特别是在神经网络做分类问题时，也经常使用交叉熵作为损失函数，此外，由于交叉熵涉及到计算每个类别的概率，所以交叉熵几乎每次都和**sigmoid(或softmax)函数**一起出现。

我们用神经网络最后一层输出的情况，来看一眼整个模型预测、获得损失和学习的流程：

1. 神经网络最后一层得到每个类别的得分**scores（也叫logits）**；
2. 该得分经过**sigmoid(或softmax)函数**获得概率输出；
3. 模型预测的类别概率输出与真实类别的one hot形式进行交叉熵损失函数的计算。

学习任务分为二分类和多分类情况，我们分别讨论这两种情况的学习过程。

## 3.1 二分类情况

![](https://pic1.zhimg.com/80/v2-d44fea1bda9338eaabf8e96df099981c_1440w.jpg)

二分类交叉熵损失函数学习过程

如上图所示，求导过程可分成三个子过程，即拆成三项偏导的乘积：

![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+L_i%7D%7B%5Cpartial+w_i%7D%3D%5Cfrac%7B1%7D%7BN%7D%5Cfrac%7B%5Cpartial+L_i%7D%7B%5Cpartial+w_i%7D%3D%5Cfrac%7B1%7D%7BN%7D%5Cfrac%7B%5Cpartial+L_i%7D%7B%5Cpartial+p_i%7D%5Ccdot+%5Cfrac%7B%5Cpartial+p_i%7D%7B%5Cpartial+s_i%7D%5Ccdot+%5Cfrac%7B%5Cpartial+s_i%7D%7B%5Cpartial+w_i%7D%5C%5C)

## 3.1.1 计算第一项： ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+L_i%7D%7B%5Cpartial+p_i%7D)

![[公式]](https://www.zhihu.com/equation?tex=L_i+%3D+-%5By_i%5Ccdot+log%28p_i%29+%2B+%281-y_i%29%5Ccdot+log%281-p_i%29%5D+%5C%5C)

- ![[公式]](https://www.zhihu.com/equation?tex=p_i) 表示样本 ![[公式]](https://www.zhihu.com/equation?tex=i) 预测为正类的概率

- ![[公式]](https://www.zhihu.com/equation?tex=y_i) 为符号函数，样本 ![[公式]](https://www.zhihu.com/equation?tex=i) 为正类时取 ![[公式]](https://www.zhihu.com/equation?tex=1) ，否则取 ![[公式]](https://www.zhihu.com/equation?tex=0)

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+%5Cfrac%7B%5Cpartial+L_i%7D%7B%5Cpartial+p_i%7D+%26%3D%5Cfrac%7B%5Cpartial+-%5By_i%5Ccdot+log%28p_i%29+%2B+%281-y_i%29%5Ccdot+log%281-p_i%29%5D%7D%7B%5Cpartial+p_i%7D%5C%5C+%26%3D+-%5Cfrac%7By_i%7D%7Bp_i%7D-%5B%281-y_i%29%5Ccdot+%5Cfrac%7B1%7D%7B1-p_i%7D%5Ccdot+%28-1%29%5D+%5C%5C++%26%3D+-%5Cfrac%7By_i%7D%7Bp_i%7D%2B%5Cfrac%7B1-y_i%7D%7B1-p_i%7D+%5C%5C+%5Cend%7Baligned%7D+%5C%5C)

## 3.1.2 计算第二项： ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+p_i%7D%7B%5Cpartial+s_i%7D+)

这一项要计算的是sigmoid函数对于score的导数，我们先回顾一下sigmoid函数和分数求导的公式：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D++%5Cfrac%7B%5Cpartial+p_i%7D%7B%5Cpartial+s_i%7D+%26%3D+%5Cfrac%7B%28e%5E%7Bs_i%7D%29%27%5Ccdot+%281%2Be%5E%7Bs_i%7D%29-e%5E%7Bs_i%7D%5Ccdot+%281%2Be%5E%7Bs_i%7D%29%27%7D%7B%281%2Be%5E%7Bs_i%7D%29%5E2%7D+%5C%5C++%26%3D+%5Cfrac%7Be%5E%7Bs_i%7D%5Ccdot+%281%2Be%5E%7Bs_i%7D%29-e%5E%7Bs_i%7D%5Ccdot+e%5E%7Bs_i%7D%7D%7B%281%2Be%5E%7Bs_i%7D%29%5E2%7D+%5C%5C++%26%3D+%5Cfrac%7Be%5E%7Bs_i%7D%7D%7B%281%2Be%5E%7Bs_i%7D%29%5E2%7D+%5C%5C++%26%3D+%5Cfrac%7Be%5E%7Bs_i%7D%7D%7B1%2Be%5E%7Bs_i%7D%7D%5Ccdot+%5Cfrac%7B1%7D%7B1%2Be%5E%7Bs_i%7D%7D+%5C%5C++%26%3D+%5Csigma%28s_i%29%5Ccdot+%5B1-%5Csigma%28s_i%29%5D+%5C%5C+%5Cend%7Baligned%7D+%5C%5C)

## 3.1.3 计算第三项： ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+s_i%7D%7B%5Cpartial+w_i+%5C%5C%7D)

一般来说，scores是输入的线性函数作用的结果，所以有：
![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+s_i%7D%7B%5Cpartial+w_i%7D%3Dx_i+%5C%5C)

## 3.1.4 计算结果 ![[公式]](https://www.zhihu.com/equation?tex=%5Cfrac%7B%5Cpartial+L_i%7D%7B%5Cpartial+w_i%7D)

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D++%5Cfrac%7B%5Cpartial+L_i%7D%7B%5Cpartial+w_i%7D+%26%3D+%5Cfrac%7B%5Cpartial+L_i%7D%7B%5Cpartial+p_i%7D%5Ccdot+%5Cfrac%7B%5Cpartial+p_i%7D%7B%5Cpartial+s_i%7D%5Ccdot+%5Cfrac%7B%5Cpartial+s_i%7D%7B%5Cpartial+w_i%7D+%5C%5C++%26%3D+%5B-%5Cfrac%7By_i%7D%7Bp_i%7D%2B%5Cfrac%7B1-y_i%7D%7B1-p_i%7D%5D+%5Ccdot+%5Csigma%28s_i%29%5Ccdot+%5B1-%5Csigma%28s_i%29%5D%5Ccdot+x_i+%5C%5C++%26%3D+%5B-%5Cfrac%7By_i%7D%7B%5Csigma%28s_i%29%7D%2B%5Cfrac%7B1-y_i%7D%7B1-%5Csigma%28s_i%29%7D%5D+%5Ccdot+%5Csigma%28s_i%29%5Ccdot+%5B1-%5Csigma%28s_i%29%5D%5Ccdot+x_i+%5C%5C++%26%3D+%5B-%5Cfrac%7By_i%7D%7B%5Csigma%28s_i%29%7D%5Ccdot+%5Csigma%28s_i%29%5Ccdot+%281-%5Csigma%28s_i%29%29%2B%5Cfrac%7B1-y_i%7D%7B1-%5Csigma%28s_i%29%7D%5Ccdot+%5Csigma%28s_i%29%5Ccdot+%281-%5Csigma%28s_i%29%29%5D%5Ccdot+x_i+%5C%5C++%26%3D+%5B-y_i%2By_i%5Ccdot+%5Csigma%28s_i%29%2B%5Csigma%28s_i%29-y_i%5Ccdot+%5Csigma%28s_i%29%5D%5Ccdot+x_i+%5C%5C++%26%3D+%5B%5Csigma%28s_i%29-y_i%5D%5Ccdot+x_i+%5C%5C+%5Cend%7Baligned%7D+%5C%5C)

可以看到，我们得到了一个非常漂亮的结果，所以，使用交叉熵损失函数，不仅可以很好的衡量模型的效果，又可以很容易的的进行求导计算。

## 3.2 多分类情况

待整理

## 4. 优缺点

## 4.1 优点

在用梯度下降法做参数更新的时候，模型学习的速度取决于两个值：一、**学习率**；二、**偏导值**。其中，学习率是我们需要设置的超参数，所以我们重点关注偏导值。从上面的式子中，我们发现，偏导值的大小取决于 ![[公式]](https://www.zhihu.com/equation?tex=x_i) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5B%5Csigma%28s%29-y%5D) ，我们重点关注后者，后者的大小值反映了我们模型的错误程度，该值越大，说明模型效果越差，但是该值越大同时也会使得偏导值越大，从而模型学习速度更快。所以，使用逻辑函数得到概率，并结合交叉熵当损失函数时，在模型效果差的时候学习速度比较快，在模型效果好的时候学习速度变慢。

## 4.2 缺点

Deng [4]在2019年提出了ArcFace Loss，并在论文里说了Softmax Loss的两个缺点：1、随着分类数目的增大，分类层的线性变化矩阵参数也随着增大；2、对于封闭集分类问题，学习到的特征是可分离的，但对于开放集人脸识别问题，所学特征却没有足够的区分性。对于人脸识别问题，首先人脸数目(对应分类数目)是很多的，而且会不断有新的人脸进来，不是一个封闭集分类问题。

另外，sigmoid(softmax)+cross-entropy loss 擅长于学习类间的信息，因为它采用了类间竞争机制，它只关心对于正确标签预测概率的准确性，忽略了其他非正确标签的差异，导致学习到的特征比较散。基于这个问题的优化有很多，比如对softmax进行改进，如L-Softmax、SM-Softmax、AM-Softmax等。
