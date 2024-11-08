# 数据预处理算子优化MOAL

## 简介
当前已提供CPU侧算子的优化，包括`ToTensor`、`Normalize`，该两个算子均已适配与OpenSora训练预处理阶段，经过验证，在高分辨场景下性能均可以得到显著提升

## 算子优化介绍
- ToTensor 
  - 实现方案：将输入的int8类型数据转换为float格式，并对数组进行统一除以255。
  - 优化方案：使用c++多线程并发实现，避免效率过低的除法的影响，使用了AMD LD3指令优化，增加内存连续性整理等手段进行性能加速。
  - 调用方式：cpu侧数据类型为`uint8`在初始化原有接口`torchvision.transforms.ToTensor`后调用时，默认使用优化方案
  

- Normalize
  - 实现方案：对输入的图像数据进行归一化操作。
  - 优化方案：使用c++多线程并发替换原始python算子，实现性能加速。
  - 调用方式：cpu侧数据类型为`float`在初始化原有接口`torchvision.transforms.Normalize`后调用时，默认使用优化方案
