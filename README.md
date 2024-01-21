# YOLO 检测 + CNN分类



## 环境部署

首先需安装 python>=3.10.2，然后将项目移至全英文路径下

进入项目路径打开cmd/bash，根据以下命令创建并激活环境 

```bash
<Windows! -->
python -m venv my_env
my_env\Scripts\activate 
python -m pip install --upgrade pip

<Linux! -->
python3 -m venv my_env
source my_env/bin/activate 
python -m pip install --upgrade pip
```

然后安装torch>=2.1.1,torchaudio>=2.1.1 torchvision>=0.16.1

在有nvidia服务的设备上，使用以下命令安装

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

在没有nvidia服务的设备上，使用以下命令安装

```bash
pip3 install torch torchvision torchaudio
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
cd Tensorrt
pip install tensorrt-8.6.1-cp310-none-win_amd64.whl   <根据操作系统与python版本选择对应wheel包安装>
```



安装后可使用以下命令依次查看torch，cuda、cudnn以及tensorrt的版本

```bash
python -c "import torch;print(torch.__version__);print(torch.version.cuda);print(torch.backends.cudnn.version())"
python -c "import tensorrt;print(tensorrt.__version__)"
```



安装其他环境依赖
    pip install -r requirements.txt



安装torch2trt

```bash
cd Torch2trt
python setup.py install
```







# 检测模型训练

## 数据集准备

将标注好的数据集放到y_data/xml_data中，参考

```
--y_data
    --xml_data
        --Annotions
            --img1.xml
            --img2.xml
        --Images
            --img1.jpg
            --img2.jpg
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

将图片放到c_data/i_database 中，运行以下程序进行检测

```bash
python detect.py
```

随后运行训练程序

```bash
python c_train.py --model googlenet
```

或双进程训练

```bash
python c_process.py
```
