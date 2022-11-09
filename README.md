# resnet50-cifar10
本文将介绍基于Pytorch，使用经典CNN网络ResNet50训练CIFAR10数据集。

## 一、数据集
### 1 `CIFAR10`数据集说明
CIFAR10数据集共有60000张彩色图像，这些图像是`32*32`，分为`10`个类，每类`6000`张图。这里面有`50000`张用于训练，构成了5个训练批，每一批10000张图；另外`10000`用于测试，单独构成一批。

### 2 数据集增强
大型数据集是成功应用深度神经网络的先决条件。 图像增广在对训练图像进行一系列的随机变化之后，生成相似但不同的训练样本，从而扩大了训练集的规模。 此外，应用图像增广的原因是，随机改变训练样本可以减少模型对某些属性的依赖，从而提高模型的泛化能力。
#### (1)图像增广

#### a.随机翻转
左右、上下翻转图像通常不会改变对象的类别。
本文用到的方法如下：
```python
transforms.RandomHorizontalFlip()
```

#### b.裁剪
可以通过对图像进行随机裁剪，使物体以不同的比例出现在图像的不同位置。 这也可以降低模型对目标位置的敏感性。
本文用到的方法如下，随机裁剪一个面积为原始面积`10%`到`100%`的区域，并在该区域的宽高比从`0.5`到`2`之间随机取值：
```python
shape_aug = torchvision.transforms.RandomResizedCrop(
    (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
apply(img, shape_aug)
```
![1061667985069_ pic](https://user-images.githubusercontent.com/63033807/200788764-d6f1f9aa-4d8b-4b97-88e8-42bf9bbb22a5.jpg)

#### c.颜色改变
改变图像颜色的四个方面：亮度、对比度、饱和度和色调。通常来说，这些操作不会改变图像的类别，但能增强模型分类的鲁棒性。

#### (2)图像增强
#### a.填充后随机裁剪
本文将尺寸为`32x32`的图像填充为`40x40`，然后随机裁剪成`32x32`。
```python
transforms.RandomCrop(32, padding=4)
```
### 3 数据集展示
数据集占比例展示
```python
'frog': 		5000			1000
'truck': 		5000			1000
'deer': 		5000			1000
'automobile':   	5000			1000
'bird': 		5000			1000
'horse': 		5000			1000
'ship': 		5000			1000
'cat': 			5000			1000
'dog': 			5000			1000
'airplane': 	        5000			1000
```
![1051667983609_ pic](https://user-images.githubusercontent.com/63033807/200782893-8bfe2a60-57da-4012-ba50-27a59538398b.jpg)

## 二、模型构建
### 1 `ResNet50`
为了解决叠加layers而产生的退化问题，ResNet网络的作者提出了将堆叠的几层layer称之为一个`block`，对于某个block，其可以拟合的函数为F(x)，如果期望的潜在映射为H(x)，与其让F(x)直接学习潜在的映射，不如去学习残差H(x)−x,即F(x)=H(x)−x，这样原本的前向路径上就变成了F(x)+x，用F(x)+x来拟合H(x)。作者认为这样可能更易于优化，因为相比于让F(x)学习成恒等映射，让F(X)学习成0要更加容易——后者通过L2正则就可以轻松实现。这样，对于冗余的block，只需F(x)→0就可以得到恒等映射，性能不减。
#### a.Residual Block的两种block
残差路径可以大致分成2种，一种有`bottleneck`结构，用于先降维再升维，主要出于降低计算复杂度的现实考虑，称之为`“bottleneck block”`，另一种没有bottleneck结构，称之为`“basic block”`。

`BasicBlock`的代码实现：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
   
`	expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```
`Bottleneck`的代码实现：
```python
class Bottleneck(nn.Module):
    
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```

#### b.Residual Block的两种shortcut
`shortcut`路径大致也可以分成2种，取决于残差路径是否改变了feature map数量和尺寸，一种是将输入原封不动地输出，另一种则需要经过1×1卷积来升维or/and降采样，主要作用是将输出与F(x)路径的输出保持shape一致，对网络性能的提升并不明显。

#### c.ResNet网络结构
`ResNet50`有四组大block，每组分别是3、4、6、3个小block，每个小block里面有三个卷积，另外这个网络的最开始有一个单独的卷积层，因此是：`（3+4+6+3）*3+1=49`，最后后又一个全连接层，因而一共50层。
`ResNet`网络的代码实现如下：
```python
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])
```
### 2 模型训练
#### a.训练与测试
训练函数代码如下：
```python
def train(epoch):
    net.train()
    epoch_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_dl):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()* inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = correct / total
    loss = epo
```
测试函数代码如下：
```python
def test(epoch):
    global best_acc
    net.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_dl):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            epoch_loss += loss.item()* inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = correct / total
    loss = epoch_loss / total

    print('test_loss: %.4f test_acc: %.4f '%(loss, acc), end=' ' )
    return {'loss': loss, 'acc': acc}
```
训练与测试部分代码如下：
```python
train_info = {'loss': [], 'acc': []}
test_info = {'loss': [], 'acc': []}
for epoch in range(61):
    time1 = time.time()
    d_train = train(epoch)
    d_test = test(epoch)
    scheduler.step()
    print("%.4ss"%(time.time() - time1), end='\n')
    for k in train_info.keys():
        train_info[k].append(d_train[k])
        test_info[k].append(d_test[k])
```
#### b.关键参数定义
学习率lr的初始值设定为`0.1`，由于本文的实验共训练60epoch，因此利用torch.optim.lr_scheduler.CosineAnnealingLR函数将学习率lr变化的1/2周期设定为`60`。
代码如下：
```python
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
args = parser.parse_args(args=[])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = ResNet50()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
```

## 三、结果展示
经过`60`代的训练，模型在训练集上的acc达到`97.81%`，在测试集上的acc达到`91.13%`。
![result](https://user-images.githubusercontent.com/63033807/200789433-88bd6568-5690-4b6b-ac04-3938639b3b49.jpeg)

以下为最后`10`个epoch的训练结果：
```python
epoches: 50 train_loss: 0.1613 train_acc: 0.9437  --> test_loss: 0.3568 test_acc: 0.8886  73.6s
epoches: 51 train_loss: 0.1445 train_acc: 0.9496  --> test_loss: 0.3252 test_acc: 0.8972  73.6s
epoches: 52 train_loss: 0.1303 train_acc: 0.9549  --> test_loss: 0.3224 test_acc: 0.9002  73.5s
epoches: 53 train_loss: 0.1125 train_acc: 0.9614  --> test_loss: 0.3165 test_acc: 0.9013  73.5s
epoches: 54 train_loss: 0.0976 train_acc: 0.9667  --> test_loss: 0.3100 test_acc: 0.9073  73.5s
epoches: 55 train_loss: 0.0871 train_acc: 0.9709  --> test_loss: 0.3152 test_acc: 0.9072  73.5s
epoches: 56 train_loss: 0.0795 train_acc: 0.9733  --> test_loss: 0.3089 test_acc: 0.9092  73.5s
epoches: 57 train_loss: 0.0731 train_acc: 0.9754  --> test_loss: 0.3033 test_acc: 0.9114  73.5s
epoches: 58 train_loss: 0.0719 train_acc: 0.9759  --> test_loss: 0.3004 test_acc: 0.9106  73.5s
epoches: 59 train_loss: 0.0679 train_acc: 0.9784  --> test_loss: 0.3028 test_acc: 0.9111  73.5s
epoches: 60 train_loss: 0.0678 train_acc: 0.9781  --> test_loss: 0.3027 test_acc: 0.9113  73.5s
```
