<div align="center">
    <h2>
        RS-Mamba for Large Remote Sensing Image Dense Prediction
    </h2>
</div>
<br>

<div align="center">
  <img src="resources/RS-Mamba.png" width="800"/>
</div>
<br>
<div align="center">
  <!-- <a href="https://kychen.me/RSMamba">
    <span style="font-size: 20px; ">项目主页</span>
  </a>
       -->
  <a href="http://arxiv.org/abs/2404.02668">
    <span style="font-size: 20px; ">arXiv</span>
  </a>
      
  <a href="resources/RS-Mamba.pdf">
    <span style="font-size: 20px; ">PDF</span>
  </a>
  <!--     
  <a href="https://huggingface.co/spaces/KyanChen/RSMamba">
    <span style="font-size: 20px; ">HFSpace</span>
  </a> -->
</div>
<br>
<br>

[![GitHub stars](https://badgen.net/github/stars/walking-shadow/Official_Remote_Sensing_Mamba)](https://github.com/walking-shadow/Official_Remote_Sensing_Mamba)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2404.02668-b31b1b.svg)](http://arxiv.org/abs/2404.02668)

<!-- [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/KyanChen/RSMamba) -->

<!-- <br>
<br> -->

<div align="center">

[English](README.md) | 简体中文

</div>

## 简介

本项目仓库是论文 [RS-Mamba for Large Remote Sensing Image Dense Prediction](http://arxiv.org/abs/2404.02668) 的代码实现，在 [VMamba](https://github.com/MzeroMiko/VMamba) 项目环境的基础上进行开发。

如果你觉得本项目对你有帮助，请给我们一个 star ⭐️。

<details open>
<summary>主要贡献</summary>

- 首次将状态空间模型（SSM）引入遥感密集预测任务，RSM在具有全局有效感受野的同时具有线性复杂度，从而能够处理保留了丰富上下文信息的大遥感图像
·
- 针对遥感图像的空间特征分布在多个方向的特点，RSM通过在多个方向对图像进行选择性扫描，从而能够在多个方向上进行全局建模，并提取出多个方向的大尺度空间特征
·
- 语义分割和变化检测任务上的实验证明了，RSM在使用最简单的模型架构和训练方法的情况下，仍然能够达到SOTA效果，具有很大的潜力

</details>

## 更新日志

🌟 **2024.03.29** 发布了 RS-Mamba 项目

🌟 **2024.04.06** 模型代码和训练框架代码已开源

## TODO

- [X] 开源模型代码
- [X] 开源训练框架

## 目录

- [简介](#简介)
- [更新日志](#更新日志)
- [TODO](#todo)
- [目录](#目录)
- [文件夹与文件说明](#文件夹与文件说明)
- [安装](#安装)
  - [环境安装](#环境安装)
- [数据集准备](#数据集准备)
  - [遥感图像语义分割数据集](#遥感图像语义分割数据集)
    - [Massachusetts Roads 数据集](#massachusetts-roads-数据集)
    - [WHU 数据集](#whu-数据集)
    - [组织方式](#组织方式)
  - [遥感图像变化检测数据集](#遥感图像变化检测数据集)
    - [WHU-CD 数据集](#whu-cd-数据集)
    - [LEIVR-CD 数据集](#leivr-cd-数据集)
    - [组织方式](#组织方式-1)
- [模型训练与推理](#模型训练与推理)
  - [语义分割模型训练与推理](#语义分割模型训练与推理)
  - [变化检测模型训练与推理](#变化检测模型训练与推理)
- [常见问题](#常见问题)
  - [1. 安装VMamba环境中的selective\_scan库出现问题](#1-安装vmamba环境中的selective_scan库出现问题)
  - [2. 运行时出现 ModuleNotFoundError: No module named 'selective\_scan\_cuda'](#2-运行时出现-modulenotfounderror-no-module-named-selective_scan_cuda)
- [引用](#引用)
- [开源许可证](#开源许可证)
- [论文解读](#论文解读)

<!-- - [致谢](#致谢) -->

- [引用](#引用)
- [开源许可证](#开源许可证)

<!-- - [联系我们](#联系我们) -->

- [论文解读](#论文解读)

## 文件夹与文件说明

`semantci_segmentation_mamba`文件夹和`change_detection_mamba`文件夹分别为进行遥感语义分割和变化检测任务的代码，它们具有相同的文件组织方式。

以`change_detection_mamba`文件夹为例，其中的`train.py`和`inference.py`分别为训练和推理代码，`rs_mamba_cd.py`为模型代码，`utils`文件夹存放着各类其他的代码文件。

`utils`文件夹中`data_loading.py`为数据加载代码文件；`dataset_process.py`为数据集处理文件，包含对变化检测数据集进行预处理的各种函数；`losses.py`为损失函数代码文件；`path_hyperparameter.py`存放着各种模型和训练的超参数，数据集的名字和模型的超参数也在其中设置；`utils.py`包含训练和验证的代码。

## 安装

### 环境安装

<details open>

**步骤 1**：按照[Vmamba项目](https://github.com/MzeroMiko/VMamba)的环境安装指示，安装好"rs_mamba"环境

**步骤 2**：运行以下命令安装依赖包

如果你只需要使用模型代码，则不需要这一步.

```shell
pip install -r requirements.txt
```

## 数据集准备

<details open>

### 遥感图像语义分割数据集

#### Massachusetts Roads 数据集

- 数据集下载地址：[Massachusetts Roads 数据集](https://www.cs.toronto.edu/~vmnih/data/)。

#### WHU 数据集

- 数据集下载地址： [WHU 数据集](http://gpcv.whu.edu.cn/data/building_dataset.html)。

#### 组织方式

你需要将数据集组织成如下的格式：

```
${DATASET_ROOT} # 数据集根目录，
├── train
    ├── image
        └── 0001.tif
        └── 0002.tif
        └── ...
    ├── label
        └── 0001.tif
        └── 0002.tif
        └── ...
├── val
    ├── image
        └──0001.tif
        └── 0002.tif
        └── ...
    ├── label
        ├── 0001.tif
        └── 0002.tif
        └── ...
├── test
    ├── image
        └── 0001.tif
        └── 0002.tif
        └── ...
    ├── label
        └── 0001.tif
        └── 0002.tif
        └── ...
```

### 遥感图像变化检测数据集

#### WHU-CD 数据集

- 数据集下载地址：[WHU-CD 数据集](http://gpcv.whu.edu.cn/data/building_dataset.html)。

#### LEIVR-CD 数据集

- 数据集下载地址： [LEVIR-CD 数据集](https://chenhao.in/LEVIR/)。

#### 组织方式

你需要将数据集组织成如下的格式：

```
${DATASET_ROOT} # 数据集根目录，
├── train
    ├── t1
        └── 0001.tif
        └── 0002.tif
        └── ...
    ├── t2
        └── 0001.tif
        └── 0002.tif
        └── ...
    ├── label
        └── 0001.tif
        └── 0002.tif
        └── ...
├── val
    ├── t1
        └── 0001.tif
        └── 0002.tif
        └── ...
    ├── t2
        └── 0001.tif
        └── 0002.tif
        └── ...
    ├── label
        ├── 0001.tif
        └── 0002.tif
        └── ...
├── test
    ├── t1
        └── 0001.tif
        └── 0002.tif
        └── ...
    ├── t2
        └── 0001.tif
        └── 0002.tif
        └── ...
    ├── label
        └── 0001.tif
        └── 0002.tif
        └── ...
```

## 模型训练与推理

模型训练和推理的所有超参数都在utils/path_hyperparameter.py文件中，每个超参数都有对应的注释进行解释。

### 语义分割模型训练与推理

首先在命令行运行以下命令跳转到语义分割文件夹下

```
cd semantic_segmentation_mamba
```

在命令行中运行下面的代码来开始训练
```
python train.py
```

如果你想在训练的时候进行调试，在命令行中运行下面的命令

```
python -m ipdb train.py
```

在命令行中运行下面的代码来开始测试或者推理

```
python inference.py
```

### 变化检测模型训练与推理

首先在命令行运行以下命令跳转到语义分割文件夹下

```
cd change_detection_mamba
```

在命令行中运行下面的代码来开始训练
```
python train.py
```

如果你想在训练的时候进行调试，在命令行中运行下面的命令

```
python -m ipdb train.py
```

在命令行中运行下面的代码来开始测试或者推理

```
python inference.py
```

<!-- ## 模型训练

### RSMamba 模型

#### Config 文件及主要参数解析

我们提供了论文中不同参数大小的 RSMamba 模型的配置文件，你可以在 [配置文件](configs/rsmamba) 文件夹中找到它们。Config 文件完全与 MMPretrain 保持一致的 API 接口及使用方法。下面我们提供了一些主要参数的解析。如果你想了解更多参数的含义，可以参考 [MMPretrain 文档](https://mmpretrain.readthedocs.io/zh-cn/latest/user_guides/config.html)。

<details>

**参数解析**：

- `work_dir`：模型训练的输出路径，一般不需要修改。
- `code_root`：代码根目录，**修改为本项目根目录的绝对路径**。
- `data_root`：数据集根目录，**修改为数据集根目录的绝对路径**。
- `batch_size`：单卡的 batch size，**需要根据显存大小进行修改**。
- `max_epochs`：最大训练轮数，一般不需要修改。
- `vis_backends/WandbVisBackend`：网络端可视化工具的配置，**打开注释后，需要在 `wandb` 官网上注册账号，可以在网络浏览器中查看训练过程中的可视化结果**。
- `model/backbone/arch`：模型的骨干网络类型，**需要根据选择的模型进行修改**，包括 `b`, `l`, `h`。
- `model/backbone/path_type`：模型的路径类型，**需要根据选择的模型进行修改**。
- `default_hooks-CheckpointHook`：模型训练过程中的检查点保存配置，一般不需要修改。
- `num_classes`：数据集的类别数，**需要根据数据集的类别数进行修改**。
- `dataset_type`：数据集的类型，**需要根据数据集的类型进行修改**。
- `resume`: 是否断点续训，一般不需要修改。
- `load_from`：模型的预训练的检查点路径，一般不需要修改。
- `data_preprocessor/mean/std`：数据预处理的均值和标准差，**需要根据数据集的均值和标准差进行修改**，一般不需要修改，参考 [Python 脚本](tools/rsmamba/get_dataset_img_meanstd.py)。

一些参数来源于 `_base_` 的继承值，你可以在 [基础配置文件](configs/rsmamba/_base_/) 文件夹中找到它们。

</details>


#### 单卡训练

```shell
python tools/train.py configs/rsmamba/name_to_config.py  # name_to_config.py 为你想要使用的配置文件
```

#### 多卡训练

```shell
sh ./tools/dist_train.sh configs/rsmamba/name_to_config.py ${GPU_NUM}  # name_to_config.py 为你想要使用的配置文件，GPU_NUM 为使用的 GPU 数量
```

### 其他图像分类模型

<details open>

如果你想使用其他图像分类模型，可以参考 [MMPretrain](https://github.com/open-mmlab/mmpretrain) 来进行模型的训练，也可以将其Config文件放入本项目的 `configs` 文件夹中，然后按照上述的方法进行训练。

</details>

## 模型测试

#### 单卡测试：

```shell
python tools/test.py configs/rsmamba/name_to_config.py ${CHECKPOINT_FILE}  # name_to_config.py 为你想要使用的配置文件，CHECKPOINT_FILE 为你想要使用的检查点文件
```

#### 多卡测试：

```shell
sh ./tools/dist_test.sh configs/rsmamba/name_to_config.py ${CHECKPOINT_FILE} ${GPU_NUM}  # name_to_config.py 为你想要使用的配置文件，CHECKPOINT_FILE 为你想要使用的检查点文件，GPU_NUM 为使用的 GPU 数量
```


## 图像预测

#### 单张图像预测：

```shell
python demo/image_demo.py ${IMAGE_FILE}  configs/rsmamba/name_to_config.py --checkpoint ${CHECKPOINT_FILE} --show-dir ${OUTPUT_DIR}  # IMAGE_FILE 为你想要预测的图像文件，name_to_config.py 为你想要使用的配置文件，CHECKPOINT_FILE 为你想要使用的检查点文件，OUTPUT_DIR 为预测结果的输出路径
```

#### 多张图像预测：

```shell
python demo/image_demo.py ${IMAGE_DIR}  configs/rsmamba/name_to_config.py --checkpoint ${CHECKPOINT_FILE} --show-dir ${OUTPUT_DIR}  # IMAGE_DIR 为你想要预测的图像文件夹，name_to_config.py 为你想要使用的配置文件，CHECKPOINT_FILE 为你想要使用的检查点文件，OUTPUT_DIR 为预测结果的输出路径
``` -->

## 常见问题

<details open>

我们在这里列出了使用时的一些常见问题及其相应的解决方案，后续如果issue中有经常出现的问题，也会在这里列出来。

### 1. 安装VMamba环境中的selective_scan库出现问题

可以参考VMamba的[issue102](https://github.com/MzeroMiko/VMamba/issues/102), [issue95](https://github.com/MzeroMiko/VMamba/issues/95), 我的做法是询问GPT4之后，得到了可行的解决方法，使用conda更新GCC即可，相关询问和回答在[这里](https://chat.openai.com/share/afa38b89-db2d-4db0-aa61-7af16b067335).

### 2. 运行时出现 ModuleNotFoundError: No module named 'selective_scan_cuda'

可以参考VMamba的[issue55](https://github.com/MzeroMiko/VMamba/issues/55), selective_scan_cuda不是必要的

</details>

## 引用

如果你在研究中使用了本项目的代码或者性能基准，请参考如下 bibtex 引用 RSMamba。

```
@misc{zhao2024rsmamba,
      title={RS-Mamba for Large Remote Sensing Image Dense Prediction}, 
      author={Sijie Zhao and Hao Chen and Xueliang Zhang and Pengfeng Xiao and Lei Bai and Wanli Ouyang},
      year={2024},
      eprint={2404.02668},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## 开源许可证

该项目采用 [Apache 2.0 开源许可证](LICENSE)。

## 论文解读

关于这篇论文的中文解读，你可以参照这篇[知乎文章](https://zhuanlan.zhihu.com/p/689797214)

关于SSM和Mamba的学习，可以参照这两篇知乎回答：[如何理解 Mamba 模型 Selective State Spaces?](https://www.zhihu.com/question/644981978/answer/3405813530), [如何理解语言模型的训练和推理复杂度?](https://www.zhihu.com/question/644981909/answer/3401898757)
