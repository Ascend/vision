# vision
<h2 id="简介md">简介</h2>

本项目开发了Torchvision Adapter插件，用于昇腾适配Torchvision框架。
<h2 id="md">前提条件</h2>

- 需完成CANN开发或运行环境的安装，具体操作请参考《CANN 软件安装指南》。
- 需完成PyTorch Adapter插件安装，具体请参考 https://gitee.com/ascend/pytorch。
- Python支持版本为3.7，PyTorch支持版本为1.8.1, Torchvision支持版本为0.9.1。
# 安装方式

## 编译安装PyTorch和昇腾插件，具体请参考 https://gitee.com/ascend/pytorch

## 下载torchvision


```
pip3 install torchvision==0.9.1
```
## 编译生成torchvision_npu插件的二进制安装包

```
# 下载master分支代码，进入插件根目录
git clone -b master https://gitee.com/ascend/vision.git
cd vision
# 编包
python setup.py bdist_wheel
```

## 安装vision/dist下的插件torchvision_npu包

```
pip install torchvision_npu-0.9.1-py3-none-any.whl
```

# 运行

## 运行环境变量

设置环境变量脚本，例如：

```
# **指的CANN包的安装目录，CANN-xx指的是版本，{arch}为架构名称。
source /**/CANN-xx/{arch}-linux/bin/setenv.bash
```
## 执行单元测试脚本

验证运行, 输出结果OK


```shell
cd test
python -m unittest discover
```

# 公开网址
在torchvision_npu的配置文件和脚本中存在的公网地址：

| 类型 | 开源代码地址| 文件名  | 公网IP地址/公网URL地址/域名/邮箱地址 | 用途说明 |
| ----- | --------- | ----------- | ------- | ------- |
| 开发引入 | 不涉及 | vision/setup.cfg | https://gitee.com/ascend/vision | 用于打包whl的url入参 |

**Torchvision Adapter插件的适配方案见[适配指导](docs/适配指导.md)。**

