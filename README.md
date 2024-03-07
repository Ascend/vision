# Torchvision Adapter

-   [简介]()
-   [安装]()
-   [快速上手]()
-   [特性介绍]()
-   [安全声明]()


# 简介

本项目开发了Torchvision Adapter插件，用于昇腾适配Torchvision框架。
目前该适配框架增加了对Torchvision所提供的常用算子的支持，后续将会提供基于cv2和基于昇腾NPU的图像处理加速后端以加速图像处理。

# 安装

**前提条件**
- 需完成CANN开发或运行环境的安装，具体操作请参考《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/601/envdeployment/instg/instg_000002.html)》。
- 需完成PyTorch Adapter插件安装，具体请参考 https://gitee.com/ascend/pytorch 。
- Python支持版本为3.7.5，PyTorch支持版本为1.11.0, Torchvision支持版本为0.12.0。
- 需在NPU设备上基于Torchvision源码编译并安装版本为0.12.0的Torchvision wheel包。


**安装步骤**
1. 安装PyTorch和昇腾插件。

   请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》安装PyTorch和昇腾插件。

2. 编译安装Torchvision。

   按照以下命令进行编译安装。

   ```
    git clone https://github.com/pytorch/vision.git
    cd vision
    git checkout v0.12.0
    # 编包
    python setup.py bdist_wheel
    # 安装
    cd dist
    pip3 install torchvision-0.12.*.whl
   ```

3. 编译安装Torchvision Adapter插件。

   按照以下命令进行编译安装。

   ```
    # 下载master分支代码，进入插件根目录
    git clone -b v0.12.0 https://gitee.com/ascend/vision.git vision_npu
    cd vision_npu
    # 安装依赖库
    pip3 install -r requirement.txt
    # 编包
    python setup.py bdist_wheel
    # 安装
    cd dist
    pip install torchvision_npu-0.12.*.whl
   ```

# 快速上手

## 运行环境变量

   - 设置环境变量脚本，例如：

   ```
    # **指的CANN包的安装目录，CANN-xx指的是版本，{arch}为架构名称。
    source /**/CANN-xx/{arch}-linux/bin/setenv.bash
   ```

## NPU 适配。

   以Torchvision的`torchvision.ops.nms`算子为例，在cuda/cpu环境中，该算子通过如下方法进行调用：

   ``` python
    # 算子的cuda/cpu版本调用
    import torch
    import torchvision
    
    ...
    torchvision.ops.nms(boxes, scores, iou_threshold) # boxes 和 scores 为 CPU/CUDA Tensor
   ```

   经过安装Torchvision Adapter插件之后，只需增加`import torchvision_npu`则可按照原生方式调用Torchvision算子。

   ```python
    # 算子的npu版本调用
    import torch
    import torch_npu
    import torchvision
    import torchvision_npu
    
    ...
    torchvision.ops.nms(boxes, scores, iou_threshold) # boxes 和 scores 为 NPU Tensor
   ```

# 特性介绍

## NPU算子支持原生算子列表
   对于torchvision中的原生算子支持情况如表3所示。

   **表 3**  NPU支持的原生算子列表

   | 算子            | 是否支持 |
   |---------------|------|
   | nms           | √    |
   | deform_conv2d | √    |
   | ps_roi_align  | -    |
   | ps_roi_pool   | -    |
   | roi_align     | √    |
   | roi_pool      | √    |

**Torchvision Adapter插件的适配方案见[适配指导](docs/适配指导.md)。**

# 安全声明

## 系统安全加固
推荐运行环境ASLR级别为2，大部分系统默认为2。
   ```
   查看ASLR级别
    cat /proc/sys/kernel/randomize_va_space
   设置ASLR级别为2
    echo 2 > /proc/sys/kernel/randomize_va_space
   ```
## 运行用户建议
出于安全性及权限最小化角度考虑，建议使用非root等管理员类型账户使用torchvision_npu

## 文件权限控制

1. 建议用户在主机（包括宿主机）及容器中设置运行系统umask值为0027及以上，保障新增文件夹默认最高权限为750，新增文件默认最高权限为640。
2. 建议用户对个人数据、商业资产、源文件、训练过程中保存的各类文件等敏感内容做好权限管控。涉及场景如torch_npu和torchvision_npu安装目录权限管控、多用户使用共享数据集权限管控、profiling生成性能分析数据权限管控等场景，管控权限可参考表2进行设置。


**表 4**  文件（夹）各场景权限管控推荐最大值

| 类型           | linux权限参考最大值 |
| -------------- | ---------------  |
| 用户主目录                        |   750（rwxr-x---）            |
| 程序文件(含脚本文件、库文件等)       |   550（r-xr-x---）             |
| 程序文件目录                      |   550（r-xr-x---）            |
| 配置文件                          |  640（rw-r-----）             |
| 配置文件目录                      |   750（rwxr-x---）            |
| 日志文件(记录完毕或者已经归档)        |  440（r--r-----）             | 
| 日志文件(正在记录)                |    640（rw-r-----）           |
| 日志文件目录                      |   750（rwxr-x---）            |
| Debug文件                         |  640（rw-r-----）         |
| Debug文件目录                     |   750（rwxr-x---）  |
| 临时文件目录                      |   750（rwxr-x---）   |
| 维护升级文件目录                  |   770（rwxrwx---）    |
| 业务数据文件                      |   640（rw-r-----）    |
| 业务数据文件目录                  |   750（rwxr-x---）      |
| 密钥组件、私钥、证书、密文文件目录    |  700（rwx—----）      |
| 密钥组件、私钥、证书、加密密文        | 600（rw-------）      |
| 加解密接口、加解密脚本            |   500（r-x------）        |

## 构建安全声明

   torchvision_npu在源码安装过程中，会依赖torch,torch_npu和torchvision三方库，编译过程中会产生临时编译目录和程序文件。用户可根据需要，对源代码中的文件及文件夹进行权限管控，降低安全风险。

## 运行安全声明

   1.建议用户结合运行环境资源状况编写对应训练脚本。若训练脚本与资源状况不匹配，如数据集加载内存大小超出内存容量限制、训练脚本在本地生成数据超过磁盘空间大小等情况，可能引发错误并导致进程意外退出。
   
   2.PyTorch和torch_npu在运行异常时会退出进程并打印报错信息，属于正常现象。建议用户根据报错提示定位具体错误原因，包括通过设定算子同步执行、查看CANN日志、解析生成的Core Dump文件等方式。


## 公开网址声明
**表 5** torchvision_npu的配置文件和脚本中存在的公网地址

| 类型 | 开源代码地址| 文件名  | 公网IP地址/公网URL地址/域名/邮箱地址 | 用途说明 |
| ----- | --------- | ----------- | ------- | ------- |
| 开发引入 | 不涉及 | vision/setup.cfg | https://gitee.com/ascend/vision | 用于打包whl的url入参 |


## 公开接口声明
torchvision_npu 不对外暴露任何对外接口。为使torchvison在NPU上运行，我们通过Monkey Patch技术对torchvision原有函数的实现进行替换。用户使用原生torchvision库的接口，运行时执行torchvision_npu库中替换的函数实现。


