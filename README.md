# Torchvision Adapter

-   [简介]()
-   [安装]()
-   [快速上手]()
-   [特性介绍]()
-   [安全声明]()


# 简介

本项目开发了Torchvision Adapter插件，用于昇腾适配Torchvision框架。
目前该适配框架增加了对Torchvision所提供的常用算子的支持，提供了基于cv2和基于昇腾NPU的图像处理加速后端以加速图像处理。

# 安装

**前提条件**
- 需完成CANN开发或运行环境的安装，具体操作请参考《[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/canncommercial/601/envdeployment/instg/instg_000002.html)》。
- 需完成PyTorch Adapter插件安装，具体请参考 https://gitee.com/ascend/pytorch 。
- 昇腾软件栈需要安装的版本请参考[版本配套表](#版本配套表)
- Python支持版本为3.8.X，3.9.X，3.10.X，PyTorch支持版本为2.1.0, Torchvision支持版本为0.16.0。

**安装步骤**

1. 编译安装Torchvision。

   方法1：源码编译安装
   按照以下命令进行编译安装。
   ```shell
    git clone https://github.com/pytorch/vision.git
    cd vision
    git checkout v0.16.0
    # 编包
    python setup.py bdist_wheel
    # 安装
    cd dist
    pip3 install torchvision-0.16.*.whl
   ```
   方法2：pip安装
   ```shell
   pip install torchvision==0.16.0
   ```

2. 编译安装Torchvision Adapter插件。
   使用最新版本，可拉取对应分支最新代码编译安装，稳定版本可以切换到对应分支的tag, 参考[版本配套表](#版本配套表)。

   按照以下命令进行编译安装。

   ```shell
    # 下载Torchvision Adapter代码，进入插件根目录
    git clone https://gitee.com/ascend/vision.git vision_npu
    cd vision_npu
    git checkout v0.16.0-6.0.rc2
    # 安装依赖库
    pip3 install -r requirement.txt
    # 初始化CANN环境变量
    source /usr/local/Ascend/ascend-toolkit/set_env.sh # Default path, change it if needed.
    # 编包
    python setup.py bdist_wheel
    # 安装
    cd dist
    pip install torchvision_npu-0.16.*.whl
   ```

# 快速上手

## 运行环境变量

   - 运行以下命令初始化CANN环境变量

   ```
    # Default path, change it if needed.
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```
   注：docker中运行使用dvpp功能，需要映射`/dev/dvpp_cmdlist`设备文件

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
## 使用cv2图像处理后端

1. Opencv-python版本推荐。推荐使用opencv-python=4.6.0。

   ```python
    pip3 install opencv-python==4.6.0.66
   ```



2. 脚本适配。

   通过以下方式使能Opencv加速，在导入torchvision相关包前导入torchvision_npu包，在构造dataset前设置图像处理后端为cv2：

   ```python
   # 使能cv2图像处理后端
   
    ...
    import torchvision
    import torchvision_npu # 导入torchvision_npu包
    import torchvision.datasets as datasets
    ...
    torchvision.set_image_backend('cv2') # 设置图像处理后端为cv2
    ...
    train_dataset = torchvision.datasets.ImageFolder(...)
    ...
   ```

   

3. cv2算子适配原则。

   - transforms方法实际调用pillow算子以及tensor算子，cv2算子调用接口与pillow算子调用接口保持一致。

   - **cv2算子只支持numpy.ndarray作为入参，否则会直接抛出类型异常。**

     ```python
     TypeError(
         "Using cv2 backend, the image data type should be numpy.ndarray. Unexpected type {}".format(type(img)))
     ```

   - **cv2算子不支持pillow算子的BOX、HAMMING插值方式，会直接抛类型异常。**

     由于pillow算子共有6种插值方式分别是NEAREST、BILINEAR、BICUBIC、BOX、HAMMING、LANCZOS，但cv2算子支持5种插值方式NEAREST、LINEAR 、AREA、CUBIC、LANCZOS4，pillow算子的BOX、HAMMING插值方式存在无法映射cv2算子实现，此时使用cv2图像处理后端会直接抛出TypeError。

     ```python
     TypeError("Opencv does not support box and hamming interpolation")
     ```

   - cv2算子插值底层实现和pillow插值底层实现略有差异，存在图像处理结果差异，因此由插值方式导致的图像处理结果不一致情况为正常现象，通常两者结果以余弦相似度计算，结果近似在99%以内。


4. cv2算子支持列表以及性能加速情况。
   
   单算子实验结果在arm架构的Atlas训练系列产品上获得，单算子实验的cv2算子输入为np.ndarray，pillow算子输入为Image.Image。cv2算子支持列表见表1。

   **表 1**  cv2算子支持列表
      
     | ops               | 处理结果是否和pillow完全一致       | cv2单算子FPS | pillow单算子FPS | 加速比      |
   |-------------------------| -------------------------------------- | ------------ | --------------- | ----------- |
   | to_pil_image      | √（只接受tensor或np.ndarray） | -            | -               |             |
   | pil_to_tensor     | √                       | 753          | **2244**        | -198%       |
   | to_tensor         | √                       | **259**      | 240             | **7.9%**    |
   | normalize         | √（只接受tensor输入）          | -            | -               |             |
   | hflip             | √                       | **4629**     | 4230            | **9.43%**   |
   | resized_crop      | 插值底层实现有差异               | **1096**     | 445             | **146.29%** |
   | vflip             | √                       | **8795**     | 6587            | **33.52%**  |
   | resize            | 插值底层实现有差异               | **1086**     | 504             | **115.48%** |
   | crop              | √                       | **10928**    | 6743            | **62.06%**  |
   | center_crop       | **√**                   | **19267**    | 9606            | **100.57%** |
   | pad               | √                       | **3394**     | 1310            | **159.08%** |
   | rotate            | 插值底层实现有差异               | **1597**     | 1346            | **18.65%**  |
   | affine            | 插值底层实现差异，仿射矩阵获取也有差异     | **1604**     | 1287            | **24.64%**  |
   | invert            | √                       | **8110**     | 2852            | **184.36%** |
   | perspective       | 插值底层实现有差异               | **674**      | 288             | **134.03%** |
   | adjust_brightness | √                       | **1174**     | 510             | **130.20%** |
   | adjust_contrast   | √                       | **610**      | 326             | **87.12%**  |
   | adjust_saturation | **√**                   | **603**      | 385             | **56.62%**  |
   | adjust_hue        | 底层实现有差异                 | **278**      | 76              | **265.79%** |
   | posterize         | √                       | **2604**     | 2356            | **10.53%**  |
   | solarize          | √                       | **3109**     | 2710            | **14.72%**  |
   | adjust_sharpness  | 底层实现有差异                 | **314**      | 293             | **7.17%**   |
   | autocontrast      | √                       | **569**      | 540             | **5.37%**   |
   | equalize          | 底层实现有差异                 | **764**      | 590             | **29.49%**  |
   | gaussian_blur     | 底层实现有差异                 | **1190**     | 2               | **59400%**  |
   | rgb_to_grayscale  | 底层实现有差异                 | **3404**     | 710             | **379.44%** |

## 使用DVPP图像处理后端

1. 脚本适配。

   通过以下方式使能DVPP加速，在导入torchvision相关包后导入torchvision_npu包。
```python
   # 使能DVPP图像处理后端
   ...
   import torch
   import torch_npu
   import torchvision
   import torchvision_npu # 导入torchvision_npu包

   torchvision.set_image_backend('npu') # 设置图像处理后端为npu，即使能DVPP加速
   npu_input = torch.randint(0, 256, (4, 3, 480, 960), dtype=torch.uint8).npu()
   output = torchvision.transforms.functional.center_crop(npu_input, (100, 200))
   ```

2. 执行单元测试脚本。

   输出结果OK即为验证成功。
   ```
   cd test/test_npu/
   python -m unittest discover
   ```

3. DVPP支持列表

   为如下图像/视频处理方法提供了DVPP处理能力，在设置图像处理后端为npu时，使能DVPP加速。支持接口列表如下表2所示。

   **表 2**  DVPP支持功能列表

   | datasets/transforms/io      | functional       | 处理结果是否和社区接口一致 |    限制                 |
   |----------------------|------------------|--------------------|------------------------------  |
   | default_loader  |    | 输出为NPU tensor，一般与to_tensor搭配使用                        | JPEG图像分辨率: 6x4~32768x32768   |
   | ToTensor             | to_tensor        | 支持四维tensor输入，一般与default_loader搭配使用                 | 分辨率: 6x4~4096x8192     |
   | Normalize            | normalize        | √                        | 分辨率: 6x4~4096x8192     |
   | CenterCrop<br>FiveCrop<br>TenCrop | crop      | √                        | 分辨率: 6x4~32768x32768   |
   | Pad                  | pad              | √                        | 分辨率: 6x4~32768x32768<br>填充宽度支持范围[0,2048] |
   | RandomHorizontalFlip | hflip            | √                        | 分辨率: 6x4~4096x8192     |
   | RandomVerticalFlip   | vflip            | √                        | 分辨率: 6x4~4096x8192     |
   | RandomResizedCrop<br>RandomSizedCrop | resized_crop     | BILINEAR和BICUBIC: 误差+1左右。其他插值模式无法对标 | 分辨率: 6x4~32768x32768<br>输出宽超过4096时输入宽高不能小于128x16 |
   | ColorJitter          | adjust_hue       | 底层实现有差异，误差±1左右 | 分辨率: 6x4~4096x8192     |
   | ColorJitter          | adjust_contrast  | 底层实现有差异，factor在[0,1]时误差±1 | 分辨率: 6x4~4096x8192     |
   | ColorJitter          | adjust_brightness| 底层实现有差异，误差±1左右 | 分辨率: 6x4~4096x8192     |
   | ColorJitter          | adjust_saturation| 底层实现有差异，factor在[0,1]时误差±1 | 分辨率: 6x4~4096x8192     |
   | GaussianBlur         | gaussian_blur    | 底层实现有差异，误差±1左右 | 分辨率: 6x4~4096x8192<br>kernel_size只支持1、3、5 |
   | RandomAffine         | affine           | 底层实现有差异 | 分辨率: 6x4~32768x32768 |
   | RandomPerspective    | perspective      | 底层实现有差异 | 分辨率: 6x4~4096x8192 |
   | RandomRotation       | rotate           | 底层实现有差异 | 分辨率: 6x4~32768x32768 |
   | Grayscale<br>RandomGrayscale | rgb_to_grayscale | float误差±1/255, uint8误差±1 | 分辨率: 6x4~4096x8192 |
   | RandomPosterize      | posterize    | √ | 分辨率: 6x4~4096x8192 |
   | RandomSolarize       | solarize     | float误差在±0.1，uint对标 | 分辨率: 6x4~4096x8192 |
   | RandomInvert         | invert       | √ | 分辨率: 6x4~4096x8192 |
   | encode_jpeg          |              |   | 分辨率: 32x32~8192x8192<br>输出宽高需要2对齐 |


## 使用DVPP视频处理后端

1. 脚本适配。

   通过以下方式使能DVPP加速，在导入torchvision相关包后导入torchvision_npu包，并设置视频处理后端为npu：

   ```python
   # 使能DVPP视频处理后端
   ...
   import torchvision
   import torchvision_npu # 导入torchvision_npu包
   ...
   torchvision.set_video_backend('npu') # 设置视频处理后端为npu，即使能DVPP加速
   ...
   vframes, aframes, info = torchvision.io.read_video(...)
   ...
   ```

2. 执行单元测试脚本。

   输出结果OK即为验证成功。

   ```
   cd test/test_npu/
   python test_read_video.py
   ```

3. DVPP支持列表

   为如下视频处理方法提供了DVPP处理能力，在设置视频处理后端为npu时，使能DVPP加速。支持接口列表如下表3所示。

   **表 3**  DVPP支持功能列表

   | functional | 处理结果是否和pyav完全一致 | 限制                    |
   | ---------- | -------------------------- | ----------------------- |
   | read_video | 底层实现有差异，误差±3左右 | 仅支持h264/h265编码格式 |



## NPU算子支持原生算子列表

   对于torchvision中的原生算子支持情况如表3所示。

   **表 4**  NPU支持的原生算子列表

   | 算子            | 是否支持 |
   |---------------|------|
   | nms           | √    |
   | deform_conv2d | √    |
   | ps_roi_align  | -    |
   | ps_roi_pool   | -    |
   | roi_align     | √    |
   | roi_pool      | √    |

# 版本配套表

|Torchvision Adapter分支 |Torchvision Adapter Tag  | PyTorch版本   | PyTorch Extension版本    |Torchvision版本 | 驱动版本 | CANN版本|
| -----------------------|----------------------- | ------------- | ------------------------ | ------------- | -----| ---------|
| master |     v0.16.0-6.0.rc2        |     2.1.0     |   2.1.0.post6 | 0.16.0  | Ascend HDK 24.1.RC2 | CANN 8.0.RC2|
| v0.12.0-dev |     v0.12.0-6.0.rc2        |     1.11.0     |   1.11.0.post14 | 0.12.0  | Ascend HDK 24.1.RC2 | CANN 8.0.RC2|

**Torchvision Adapter适配NPU的方案见[适配指导](docs/适配指导.md)。**

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
出于安全性及权限最小化角度考虑，不建议使用root等管理员类型账户使用torchvision_npu。

## 文件权限控制

1. 建议用户在主机（包括宿主机）及容器中设置运行系统umask值为0027及以上，保障新增文件夹默认最高权限为750，新增文件默认最高权限为640。
2. 建议用户对个人数据、商业资产、源文件、训练过程中保存的各类文件等敏感内容做好权限管控。涉及场景如torch_npu和torchvision_npu安装目录权限管控、多用户使用共享数据集权限管控等场景，管控权限可参考表4进行设置。

**表 5**  文件（夹）各场景权限管控推荐最大值

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

   Torchvision_npu在源码安装过程中，会依赖torch,torch_npu和torchvision三方库，编译过程中会产生临时编译目录和程序文件。用户可根据需要，对源代码中的文件及文件夹进行权限管控，降低安全风险。

## 运行安全声明

   1.建议用户结合运行环境资源状况编写对应训练脚本。若训练脚本与资源状况不匹配，如数据集加载内存大小超出内存容量限制、训练脚本在本地生成数据超过磁盘空间大小等情况，可能引发错误并导致进程意外退出。
   
   2.Torchvision和torchvision_npu在运行异常时会退出进程并打印报错信息，属于正常现象。建议用户根据报错提示定位具体错误原因，包括通过设定算子同步执行、查看CANN日志、解析生成的Core Dump文件等方式。


## 公网地址声明
**表 6** torchvision_npu的配置文件和脚本中存在的公网地址

| 类型 | 开源代码地址| 文件名  | 公网IP地址/公网URL地址/域名/邮箱地址 | 用途说明 |
| ----- | --------- | ----------- | ------- | ------- |
| 开发引入 | 不涉及 | vision/setup.cfg | https://gitee.com/ascend/vision | 用于打包whl的url入参 |


## 公开接口声明
torchvision_npu 不对外暴露任何公开接口。为使torchvison在NPU上运行，我们通过Monkey Patch技术对torchvision原有函数的实现进行替换。用户使用原生torchvision库的接口，运行时执行torchvision_npu库中替换的函数实现。
