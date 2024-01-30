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
- Python支持版本为3.7.5，PyTorch支持版本为1.8.1, Torchvision支持版本为0.9.1。
- 需在NPU设备上基于Torchvision源码编译并安装版本为0.9.1的Torchvision wheel包。


## 安装

1. 安装PyTorch和昇腾插件。

   请参考《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》安装PyTorch和昇腾插件。

2. 编译安装Torchvision。

   按照以下命令进行编译安装。

   ```
    git clone https://github.com/pytorch/vision.git
    cd vision
    git checkout v0.9.1
    # 编包
    python setup.py bdist_wheel
    # 安装
    cd dist
    pip3 install torchvision-0.9.*.whl
   ```

3. 编译安装Torchvision Adapter插件。

   按照以下命令进行编译安装。

   ```
    # 下载master分支代码，进入插件根目录
    git clone -b master https://gitee.com/ascend/vision.git vision_npu
    cd vision_npu
    git checkout v0.9.1
    # 安装依赖库
    pip3 install -r requirement.txt
    # 编包
    python setup.py bdist_wheel
    # 安装
    cd dist
    pip install torchvision_npu-0.9.*.whl
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
    torchvision_npu.set_image_backend('cv2') # 设置图像处理后端为cv2
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
   
   单算子实验结果在arm架构的昇腾芯片910A上获得，单算子实验的cv2算子输入为np.ndarray，pillow算子输入为Image.Image。
      
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

1. 设置环境变量。

   ```
   # **指的CANN包的安装目录，CANN-xx指的是版本。
   source /**/CANN-xx/latest/bin/setenv.bash
   ```

2. 脚本适配。

   通过以下方式使能DVPP加速，在导入torchvision相关包前导入torchvision_npu包，在构造dataset前设置图像处理后端为npu：
   ```python
   # 使能DVPP图像处理后端
   ...
   import torchvision
   import torchvision_npu # 导入torchvision_npu包
   import torchvision.datasets as datasets
   ...
   torchvision_npu.set_image_backend('npu') # 设置图像处理后端为npu，即使能DVPP加速
   ...
   train_dataset = torchvision.datasets.ImageFolder(...)
   ...
   ```

   数据预处理多进程场景下，worker进程默认运行在主进程设置的deive上（如无设置默认0）。
   可通过set_accelerate_npu接口设置worker进程的device，例如：
   ``` python
   # 设置worker进程的device_id
   ...
   train_dataset = torchvision.datasets.ImageFolder(...)
   train_dataset.set_accelerate_npu(npu=1) # npu参数表示要设置的device_id
   ...
   ```

   部分网络脚本的ImageFolder中不包括ToTensor，DataLoader中collate_fn使用自定义的fast_collate，进行数据从PIL.Image.Imgae对象到torch.Tensor的转换。此时，使用DVPP加速需要关闭fast_collate使用默认的defualt_collate。
   ```python
   ''' 用户脚本中自定义的collate函数 '''
   def fast_collate(batch):
      imgs = [img[0] for img in batch]
      targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
      w = imgs[0].size[0]
      h = imgs[0].size[1]
      tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
      for i, img in enumerate(imgs):
         nump_array = np.asarray(img, dtype=np.uint8)
         if nump_array.ndim < 3:
               nump_array = np.expand_dims(nump_array, axis=-1)
         # 如果此处没有进行nump_array从(H, W, C)到(C, H, W)的转换，那么转换会放在训练中
         nump_array = np.rollaxis(nump_array, 2)
         tensor[i] += torch.from_numpy(nump_array)

      return tensor, targets
   ...
   train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
        # collate_fn=fast_collate, drop_last=True)
        drop_last=True) # 使用collate_fn的默认defualt_collate
   ...
         if 'npu' in args.device:
               # images = images.npu(non_blocking=True).permute(0, 3, 1, 2).to(torch.float).sub(mean).div(std)
               # 如果轴转换放在训练中，使用DVPP加速时需要去掉
               images = images.npu(non_blocking=True).to(torch.float).sub(mean).div(std)
               target = target.npu(non_blocking=True)
   ...
   ```

   如果脚本中有多个dataset，希望一些dataset使用DVPP加速，一些使用原生处理，只需要在对应dataset构造前设置相应的处理后端，如：
   ```python
   ...
   torchvision_npu.set_image_backend('npu') # 设置dataset1的图像处理后端为npu
   dataset1 = torchvision.datasets.ImageFolder(...)
   ...
   torchvision_npu.set_image_backend('PIL') # 设置dataset2的图像处理后端为PIL
   dataset2 = torchvision.datasets.ImageFolder(...)
   ...
   ```

4. 执行单元测试脚本。

   输出结果OK即为验证成功。
   ```
   cd test/test_npu/
   python -m unittest discover
   ```

3. 说明。

   只有通过torchvision.datasets.ImageFolder/DatasetFolder构造的dataset才可以使能DVPP加速。
   
   torchvision.transforms方法对外接口不变，只支持NCHW(N=1)格式的npu tensor作为入参，其他限制见表2。

   物理机场景下，一个device上最多支持64个用户进程，即单p数据预处理进程数最多设置63。


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


**表 2**  DVPP支持列表

| transforms           | functional       | 处理结果是否和pillow完全一致 |    限制                 |
|----------------------|------------------|--------------------|------------------------------  |
|                      | npu_loader       | √                        | 分辨率: 6x4~16384x16384   |
| ToTensor             | to_tensor        | √                        | 分辨率: 6x4~4096x8192     |
| Normalize            | normalize        | √                        | 分辨率: 6x4~4096x8192     |
| Resize               | resize           | 底层实现有差异，误差±1左右 | 分辨率: 6x4~32768x32768   |
| CenterCrop           | center_crop      | √                        | 分辨率: 6x4~32768x32768   |
| FiveCrop             | five_crop        | √                        | 分辨率: 6x4~32768x32768   |
| TenCrop              | ten_crop         | √                        | 分辨率: 6x4~32768x32768   |
| Pad                  | pad              | √                        | 分辨率: 6x4~32768x32768<br>填充值不支持负数 |
| RandomHorizontalFlip | hflip            | √                        | 分辨率: 6x4~4096x8192     |
| RandomVerticalFlip   | vflip            | √                        | 分辨率: 6x4~4096x8192     |
| RandomResizedCrop    | resized_crop     | 底层实现有差异，误差±1左右 | 分辨率: 6x4~32768x32768<br>插值模式不支持BICUBIC |
| ColorJitter          | adjust_hue       | 底层实现有差异，误差±1左右 | 分辨率: 6x4~4096x8192     |
| ColorJitter          | adjust_contrast  | 底层实现有差异，factor在[0,1]时误差±1 | 分辨率: 6x4~4096x8192     |
| ColorJitter          | adjust_brightness| 底层实现有差异，误差±1左右 | 分辨率: 6x4~4096x8192     |
| ColorJitter          | adjust_saturation| 底层实现有差异，factor在[0,1]时误差±1 | 分辨率: 6x4~4096x8192     |
| GaussianBlur         | gaussian_blur    | 底层实现有差异，误差±1左右 | 分辨率: 6x4~4096x8192<br>kernel_size只支持1、3、5 |

### 接口说明
为使torchvison在NPU上运行，我们通过Monkey Patch技术对torchvision原有函数的实现进行替换。用户使用原生torchvision库的接口，运行时执行torchvision_npu库中替换的函数实现。

## 公开网址
**表 3** torchvision_npu的配置文件和脚本中存在的公网地址

| 类型 | 开源代码地址| 文件名  | 公网IP地址/公网URL地址/域名/邮箱地址 | 用途说明 |
| ----- | --------- | ----------- | ------- | ------- |
| 开发引入 | 不涉及 | vision/setup.cfg | https://gitee.com/ascend/vision | 用于打包whl的url入参 |

**Torchvision Adapter插件的适配方案见[适配指导](docs/适配指导.md)。**

