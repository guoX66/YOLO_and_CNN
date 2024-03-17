# YOLO 检测 + CNN分类

 本项目结合YOLO检测和CNN分类



## CNN图像分类使用子项目地址：[guoX66/CNN_IC (github.com)](https://github.com/guoX66/CNN_IC)

# 一、环境部署

## 项目拉取

用git命令把子模块也拉取下来：

```bash
git clone --recurse-submodules https://github.com/guoX66/YOLO_and_CNN.git
```

或将子项目下载后放入本项目中

## 环境部署

按子项目要求配置好环境后，再安装本项目依赖：

```bash
pip install -r requirements.txt
```

若使用openvino框架进行推理，请使用以下命令安装环境

```bash
pip install openvino-dev==2022.3.1
```

若使用TensorRt框架进行推理，请确保

tensorrt>=8.6.1,cuda>=11.8,cudnn>=8.7

tensorrt安装包可按照torch与cuda版本在官网选择下载：[NVIDIA TensorRT Download | NVIDIA Developer](https://developer.nvidia.com/tensorrt-download)

或者按照项目已包含的包安装tensorrt，执行以下命令

```bash
cd utils/Tensorrt
pip install tensorrt-8.6.1-cp310-none-win_amd64.whl   <根据操作系统与python版本选择对应wheel包安装>
```



安装后可使用以下命令依次查看torch，cuda、cudnn以及tensorrt的版本

```bash
python -c "import torch;print(torch.__version__);print(torch.version.cuda);print(torch.backends.cudnn.version())"
python -c "import tensorrt;print(tensorrt.__version__)"
```

安装其他环境依赖

```bash
pip install -r requirements.txt
```



# 检测模型训练

## 数据集准备

将标注好的数据集放到data/xml_data中，参考 [Rhierarch/simple_YOLO (github.com)](https://github.com/Rhierarch/simple_YOLO)

```
--data
    --y_data
        --xml_data
            --Annotions
                --img1.xml
                --img2.xml
                    ...
            --Images
                --img1.jpg
                --img2.jpg
                    ...
```

随后运行标签转换程序

```bash
python y_label.py
```

## 模型训练

```bash
python y_train.py
```



# 分类模型训练

将图片放到data/database 中，运行以下程序进行检测

```bash
python detect.py  
```

进入子项目路径，按照子项目步骤进行分类训练

```bash
cd CNN_IC
```


