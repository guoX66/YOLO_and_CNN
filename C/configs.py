# coding:utf-8
import argparse
import platform

from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='googlenet')
parser.add_argument('--is_divide', type=bool, default=True)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--min_acc', type=float, default=99)
args = parser.parse_args()


class ModelInfo:
    def __init__(self):
        self.model = args.model  # 选择模型，可选googlenet,resnet18，resnet34，resnet50，resnet101, DenseNet121,DenseNet161,DenseNet201
        self.modelname = 'model-' + self.model
        self.size = [256, 256]  # 设置输入模型的图片大小
        # self.ms = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]  # 标准化设置
        self.ms = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        self.min_pr = 7 / 8  # 设置预测时各模型预测结果最多的类作为最终结果所需要的最小占比，即几票通过算通过


class TrainImg:
    def __init__(self):
        self.train_path = 'c_data/static'  # 保存图片的文件名
        self.imgpath = self.train_path + '/train'
        self.is_divide = args.is_divide  # 设置训练开始前是否拆分测试集
        self.data_path = 'c_data/database'  # 拆分前数据所在文件名
        self.t_divide_present = 0.8  # 拆分测试集比例
        self.divide_present = 0.8  # 拆分验证集比例
        os_name = str(platform.system())
        if os_name == 'Windows':
            self.batch_size = 16
        else:
            self.batch_size = 128

        self.learn_rate = 0.001  # 设置学习率
        self.step_size = 1  # 学习率衰减步长
        self.gamma = 0.95  # 学习率衰减系数，也即每个epoch学习率变为原来的0.95
        self.epoch = args.epochs  # 设置迭代次数
        self.show_mode = 'No'  # 设模型层数信息写入log中的模式:'All'  'Simple'  'No'
        self.min_acc = args.min_acc  # 设置停止训练所需的acc，单位为 %，需要开启自动测试
        self.write_process = False  # 设置是否将训练过程写入log中


class TestImg(TrainImg):
    def __init__(self):
        super().__init__()
        self.imgpath = 'c_data/static/test'  # 保存测试图片的路径名
        self.log_path = 'log-test'
