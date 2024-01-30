# Torchvision Adapter

-   [简介]()
-   [前提条件]()
-   [安装]()
-   [运行]()
-   [使用DVPP图像处理后端]()
-   [NPU算子支持列表]()


## 简介

本项目开发了Torchvision Adapter插件，用于昇腾适配Torchvision框架。
目前该适配框架增加了对Torchvision所提供的常用算子的支持，后续将会提供基于cv2和基于昇腾NPU的图像处理加速后端以加速图像处理。

## 前提条件

- 需完成CANN开发或运行环境的安装，具体操作请参考《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/601/envdeployment/instg/instg_000002.html)》。
- 需完成PyTorch Adapter插件安装，具体请参考 https://gitee.com/ascend/pytorch 。
- Python支持版本为3.7.5，PyTorch支持版本为1.11.0, Torchvision支持版本为0.12.0。
- 需在NPU设备上基于Torchvision源码编译并安装版本为0.12.0的Torchvision wheel包。


## 安装

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

## 运行

1. 运行环境变量。

   - 设置环境变量脚本，例如：

   ```
    # **指的CANN包的安装目录，CANN-xx指的是版本，{arch}为架构名称。
    source /**/CANN-xx/{arch}-linux/bin/setenv.bash
   ```
   - 推荐运行环境ASLR级别为2，大部分系统默认为2。
   ```
   查看ASLR级别
    cat /proc/sys/kernel/randomize_va_space
   设置ASLR级别为2
    echo 2 > /proc/sys/kernel/randomize_va_space
   ```

2. NPU 适配。

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


## NPU算子支持列表

**表 1**  NPU支持算子列表

| 算子            | 是否支持 |
|---------------|------|
| nms           | √    |
| deform_conv2d | √    |
| ps_roi_align  | -    |
| ps_roi_pool   | -    |
| roi_align     | √    |
| roi_pool      | √    |

### 接口说明
为使torchvison在NPU上运行，我们通过Monkey Patch技术对torchvision原有函数的实现进行替换。用户使用原生torchvision库的接口，运行时执行torchvision_npu库中替换的函数实现。

## 公开网址
**表 2** torchvision_npu的配置文件和脚本中存在的公网地址

| 类型 | 开源代码地址| 文件名  | 公网IP地址/公网URL地址/域名/邮箱地址 | 用途说明 |
| ----- | --------- | ----------- | ------- | ------- |
| 开发引入 | 不涉及 | vision/setup.cfg | https://gitee.com/ascend/vision | 用于打包whl的url入参 |

**Torchvision Adapter插件的适配方案见[适配指导](docs/适配指导.md)。**

