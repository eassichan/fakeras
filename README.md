# fakeras
Fakeras: fake keras.
深度学习框架keras的山寨版.


## 1. 项目背景
> What I cannot create, I do not understand.  —— 理查德·费曼

站在程序员的角度，想要彻底弄懂一个技术原理，最好的办法就是动手实现
一遍。

正好，日本学者斋藤康毅撰写、陆宇杰翻译的
《深度学习入门——基于Python的理论与实现》（下简称[1]书）一书也持相同的观点，fakeras
最初的构想就是来源于这本书：剖开隐晦羞涩的数学公式，放弃复杂的第三方深度学习框架，仅仅
使用numpy从代码的角度来一步一步从零实现神经网络。

另一方面，在接触到[1]书之前，本人已经尝试过使用keras，
以及阅读过弗朗索瓦·肖莱著作、张亮翻译的、主要介绍使用keras来搭建各种
类型的神经网络的《Python深度学习》（下简称[2]书）。在使用keras的过程中，最大的体验
就是居然两三行代码就把神经网络给搭起来了，很爽很方便。但是，一段时间
之后，更深的感触就是，我只会调keras的api，并不明白背后的运行原理，
然后会用keras不代表你懂神经网络。

于是，便心血来潮地模仿keras的api，根据[1]书介绍的原理来动手实现一个
猥琐版的深度学习框架——fakeras，并按照[2]书的一些实验来测试它。
当然，fakeras的开发目标自然不会是不自量力地对标tensorflow、pytorch或者mxnet等非常成熟的深度学习框架，
而仅仅是一个用来学习神经网络运行原理的玩具。



## 2. 功能特性
- 第三方依赖少：
```
# 只依赖以下三个第三方包：
numpy==1.16.1
matplotlib==3.1.0
pytest==4.6.3
```
- 模（~~chao~~）仿（~~xi~~）keras的接口：
```
# 使用keras构建网络：
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))


# 使用fakeras构建网络：
from fakeras import network as model
from fakeras import layer

network = model.NeuralNetwork()
network.add(layer.Dense(512, activation='relu', initialization='xavier', input_shape=(28 * 28,)))
network.add(layer.Dense(10, activation='softmax', initialization='xavier'))
# 不同的地方就是，定义Dense层的时候，fakeras需要显式地指明权重初始化策略，毕竟“显式优于隐式”
```

- 运行性能非常慢

由于fakeras只着眼于理解神经网络的运行原理，并没有在编码层面上使用太多的奇技淫巧（例如使用cupy或者原生的cuda加速）
来优化矩阵运算的速度，所以网络的训练巨慢无比，尤其是CNN的训练。


## 3. 安装
(1). 首先克隆fakeras源码仓库到本地目录下：
```
git clone https://github.com/eassichan/fakeras
```

(2). 切换到源码仓库：
```
cd fakeras
```

(3). 由于本项目仅仅只是用来加深对神经网络运行原理的理解的玩具项目，因此建议在开发模式下安装：
```
python setup.py develop
```


## 4. 使用
下面以深度学习的hello world项目MNIST手写数字识别来演示使用fakeras来搭建神经网络（更多示例请参考examples目录下的其他例子）：
```
import numpy as np

from fakeras.datasets.mnist import load_data
from fakeras.layer import Dense
from fakeras.network import NeuralNetwork


def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    dtype = 'float32'
    x_train = x_train.reshape(60000, 784).astype(dtype) / 255
    y_train = one_hot_encode(y_train, 10, dtype)
    x_test = x_test.reshape(10000, 784).astype(dtype) / 255
    y_test = one_hot_encode(y_test, 10, dtype)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    model = NeuralNetwork()
    model.add(Dense(512, activation='relu', initialization='xavier', input_shape=(784,)))
    model.add(Dense(10, activation='softmax', initialization='xavier'))
    model.build()
    model.compile(optimizer='rmsprop', loss_func='cce', metric='acc')
    model.fit(inputs=x_train,
              targets=y_train,
              epochs=5,
              lr=0.01,
              batch_size=128,
              use_every_sample=True,
              verbose=True)
    result = model.evaluate(x_test, y_test)
    print(result)


def one_hot_encode(dataset, feature_num, dtype='float64'):
    assert isinstance(dataset, np.ndarray)
    sample_num = len(dataset)
    dataset_encoded = np.zeros((sample_num, feature_num), dtype=dtype)
    for sidx, tidx in enumerate(dataset):  # sample index and tag index
        dataset_encoded[sidx, tidx] = 1
    return dataset_encoded


if __name__ == '__main__':
    main()
```


## 5. 目录结构
```
/fakeras
|__ examples/  # 使用示例
    |__ boston_housing/  # boston_housing数据集的示例
    |__ imdb/  # imdb电影评论数据集的示例
    |__ mnist/ # mnist手写数据识别数据集的示例
    |__ reuters/  # reuters新闻评论数据集的示例
|__ fakeras/  # 源码目录
    |__ datasets/  # 包含上述四个数据集
        |__ data/  # 原始数据集文件
        |__ boston_housing.py
        |__ imdb.py
        |__ mnist.py
        |__ reuters.py
    |__ abstract.py    # 定义抽象计算节点
    |__ activation.py  # 激活函数模块
    |__ exception.py   # 异常与异常检测模块
    |__ initialization.py  # 权重初始化策略
    |__ layer.py  # 实现神经网络中的层
    |__ loss.py   # 损失函数模块
    |__ metric.py # 监控指标模块
    |__ network.py  # 实现神经网络
    |__ optimization.py  # 优化方法模块
|__ tests/  # 单元测试
```


## 6. 开源协议
本项目遵循Apache 2.0协议.
