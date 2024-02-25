# coding:utf-8
"""
该程序包含训练需要的函数
"""
import torch
import torch.nn as nn
import torchvision.models as models
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import time
import random
import shutil
import json
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def divide_dataset(path, o_path, train_p, val_p):
    print('正在拆分数据集......')
    shutil.rmtree(o_path, ignore_errors=True)
    train_path = os.path.join(o_path, 'train')
    train_path_i = os.path.join(train_path, 'images')
    train_path_l = os.path.join(train_path, 'labels')
    val_path = os.path.join(o_path, 'val')
    val_path_i = os.path.join(val_path, 'images')
    val_path_l = os.path.join(val_path, 'labels')

    os.mkdir(o_path)
    os.mkdir(train_path)
    os.mkdir(train_path_i)
    os.mkdir(train_path_l)
    os.mkdir(val_path)
    os.mkdir(val_path_i)
    os.mkdir(val_path_l)

    image_dir = os.path.join(path, 'images')
    txt_dir = os.path.join(path, 'labels')
    dir_list = list(os.listdir(image_dir))
    true_list = []
    background_list = []
    for filename in dir_list:
        txt_filename = f'{os.path.splitext(filename)[0]}.txt'
        img_path = os.path.join(image_dir, filename)
        txt_path = os.path.join(txt_dir, txt_filename)
        if not os.path.exists(txt_path):
            background_list.append(filename)
            shutil.copy(img_path, train_path_i)
        else:
            true_list.append(filename)

    n = len(true_list)
    random.shuffle(true_list)
    train_list = true_list[:int(n * train_p / 10)]
    val_list = true_list[int(n * train_p / 10):]

    for filename in train_list:
        txt_filename = f'{os.path.splitext(filename)[0]}.txt'
        txt_path = os.path.join(txt_dir, txt_filename)
        img_path = os.path.join(image_dir, filename)
        shutil.copy(img_path, train_path_i)
        shutil.copy(txt_path, train_path_l)

    for filename in val_list:
        txt_filename = f'{os.path.splitext(filename)[0]}.txt'
        txt_path = os.path.join(txt_dir, txt_filename)
        img_path = os.path.join(image_dir, filename)
        shutil.copy(img_path, val_path_i)
        shutil.copy(txt_path, val_path_l)
    print('拆分完毕！')


def write_json(test_dict, path):
    try:
        os.mkdir('log')
    except:
        pass
    path = os.path.join('log', path)
    json_str = json.dumps(test_dict)
    with open(f'{path}.json', 'w') as json_file:
        json_file.write(json_str)


def divide_test(path, o_path, divide_num):
    print('正在拆分数据集......')
    try:
        shutil.rmtree(o_path)
    except:
        pass
    train_path = os.path.join(o_path, 'train')
    test_path = os.path.join(o_path, 'test')

    os.makedirs(o_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    for i in os.walk(path):
        for c_fold in i[1]:
            i_fold_path = os.path.join(path, c_fold)
            o_fold_path_train = os.path.join(train_path, c_fold)
            o_fold_path_test = os.path.join(test_path, c_fold)
            try:
                os.mkdir(o_fold_path_train)
            except:
                pass
            try:
                os.mkdir(o_fold_path_test)
            except:
                pass
            file_list = os.listdir(i_fold_path)
            random.shuffle(file_list)
            train_num = int(len(file_list) * divide_num)
            train_list = file_list[:train_num]
            test_list = file_list[train_num:]
            for file in train_list:
                in_path = os.path.join(i_fold_path, file)
                o_file_path_train = os.path.join(o_fold_path_train, file)
                shutil.copy(in_path, o_fold_path_train)
            for file in test_list:
                in_path = os.path.join(i_fold_path, file)
                o_file_path_test = os.path.join(o_fold_path_test, file)
                shutil.copy(in_path, o_fold_path_test)
    print('拆分完毕')


def get_label_list(imgpath):
    file_path = f'./{imgpath}'

    path_list = []

    for i in os.walk(file_path):
        path_list.append(i)

    label_dict = dict()
    label_name_list = []
    label_list = []

    for i in range(len(path_list[0][1])):
        label = path_list[0][1][i]
        label_dict[label] = path_list[i + 1][2]

    for i in label_dict.keys():
        label_list.append(i)
        for j in label_dict[i]:
            label_name_list.append([i, j])

    return label_name_list, label_dict, label_list


def make_model(modelinfo, n, path, device):
    model_ft = sele_model(modelinfo)
    layer_list = get_layers_name(model_ft)
    last_layers_name = layer_list[-1][0]
    in_features = layer_list[-1][1].in_features
    layer1 = nn.Linear(in_features, n)
    _set_module(model_ft, last_layers_name, layer1)
    model_ft.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load(path, map_location=device).items()})
    if isinstance(model_ft, torch.nn.DataParallel):
        model_ft = model_ft.module
    return model_ft


def make_train_model(modelinfo, n):
    model_ft = sele_model(modelinfo, train=True)
    layer_list = get_layers_name(model_ft)
    last_layers_name = layer_list[-1][0]
    in_features = layer_list[-1][1].in_features
    layer1 = nn.Linear(in_features, n)
    _set_module(model_ft, last_layers_name, layer1)
    return model_ft


def bar(i, t, start):
    l = 50
    f_p = i / t
    n_p = (t - i) / t
    finsh = "▓" * int(f_p * l)
    need_do = "-" * int(n_p * l)
    progress = f_p * 100
    dur = time.perf_counter() - start
    print("\r训练进度:{:^3.2f}%[{}->{}] 用时:{:.2f}s".format(progress, finsh, need_do, dur), end="")


def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


# 将最后的全连接改为对应类的输出，使输出为对应标签个数的小数，对应不同标签的置信度

# model_ft.half()#可改为半精度，加快训练速度，在这里不需要用
def get_layers(model):
    layer_list = []
    for layer in model.modules():  # 获取所有层数的名称
        layer_list.append(layer)
    return layer_list


def get_layers_name(model):
    layer_name_list = []
    for layer in model.named_modules():  # 获取所有层数的名称
        layer_name_list.append(layer)
    return layer_name_list


def show(c, model, txt_list):
    layer_list = get_layers_name(model)  # 获取模型各层信息
    if c.show_mode == 'All':
        for layers in layer_list:
            txt_list.append(str(layers) + '\r\n')

    elif c.show_mode == 'Simple':
        for layers in layer_list:
            txt_list.append(str(layers[0]) + '\r\n')


def lock(model, start, end):
    layer_list = []
    for layer in model.named_modules():  # 获取所有层数的名称
        layer_list.append(layer[0])

    need_frozen_list = layer_list[start:end]

    for module in need_frozen_list:  # 匹配并冻结对应网络层
        for param in model.named_parameters():
            if module in param[0]:
                param[1].requires_grad = False


def sele_model(Model, train=False):
    model_dict = {
        'resnet18': models.resnet18(weights=models.ResNet18_Weights.DEFAULT),  # 残差网络
        'resnet34': models.resnet34(weights=models.ResNet34_Weights.DEFAULT),
        'resnet50': models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        'resnet101': models.resnet101(weights=models.ResNet101_Weights.DEFAULT),
        'googlenet': models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT),
        'DenseNet121': models.densenet121(weights=models.DenseNet121_Weights.DEFAULT),
        'DenseNet161': models.densenet161(weights=models.DenseNet161_Weights.DEFAULT),
        'DenseNet201': models.densenet201(weights=models.DenseNet201_Weights.DEFAULT),

    }
    if train:
        return model_dict[Model.model]
    else:
        return model_dict[Model]


def get_labellist(c):
    label_name_list, _, label_list = get_label_list(c.imgpath)
    return label_name_list, label_list


def write_log(in_path, filename, txt_list):
    try:
        os.mkdir(in_path)
    except:
        pass
    path = os.path.join(in_path, filename + '.txt')
    content = ''
    for txt in txt_list:
        content += txt
    with open(path, 'w+', encoding='utf8') as f:
        f.write(content)


def add_log(txt, txt_list, is_print=True):
    if is_print:
        print(txt)
    txt_list.append(txt + '\r\n')


def train_dir(filename):
    try:
        os.mkdir('train_process')
    except:
        pass
    file_path = 'train_process/' + filename
    try:
        os.mkdir(file_path)
    except:
        pass


def make_plot(data, mode, filename, epoch):
    file_path = 'train_process/' + filename
    if mode == 'loss':
        title = 'LOSS'
        path = os.path.join(file_path, 'LOSS-' + filename)
    elif mode == 'acc':
        title = 'ACC'
        path = os.path.join(file_path, 'ACC-' + filename)
    figure(figsize=(12.8, 9.6))
    x = [i + 1 for i in range(epoch)]
    plt.plot(x, data)
    plt.rcParams['axes.unicode_minus'] = False
    plt.title(title + '-' + filename, fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(f'{path}.png')
