# coding:utf-8
import platform
import torch
from PIL import Image
from C.utils import *
from torchvision import transforms
import re
from C.configs import TrainImg, TestImg, ModelInfo
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def t_img(txt_list, model_name):
    gpus = [0, 1]
    if torch.cuda.is_available():
        torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    add_log('-' * 43 + '测试错误信息如下' + '-' * 43, txt_list)
    print()
    Train = TrainImg()
    testimgs = TestImg()
    with open("log/class_id.json", 'r', encoding='UTF-8') as f:
        class_dict = json.load(f)
    class_dict = {int(k): class_dict[k] for k in class_dict.keys()}

    testimg_path = testimgs.imgpath
    modelinfo = ModelInfo()
    transform = transforms.Compose([
        transforms.Resize([modelinfo.size[0], modelinfo.size[1]]),
        transforms.ToTensor(),
        transforms.Normalize(modelinfo.ms[0], modelinfo.ms[1])
    ])
    right_num = 0
    test_num = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_label = len(list(class_dict.keys()))
    piece = model_name.split(('-'))
    model = make_model(piece[1], n_label, model_name+'.pth', device)
    model.eval()
    os_name = str(platform.system())
    if os_name == 'Windows':
        num_workers = 0
    else:
        num_workers = 32
    model = model.to(device)  # 将模型迁移到gpu
    try:
        model = nn.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    except AssertionError:
        pass
    dataset_test = ImageFolder(testimg_path, transform=transform)  # 训练数据集
    img_list = dataset_test.imgs
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=num_workers, drop_last=False,
                                 pin_memory=True)
    for img, label in dataloader_test:
        # plt.imshow(img)
        img = img.to(device)
        output = model(img)
        # print(output)
        # print(label)
        pre = output.argmax(1)
        predict_class = class_dict[int(pre)]
        real_class = class_dict[int(label)]
        if predict_class == real_class:
            right_num += 1
        else:
            img_name = img_list[test_num][0]
            img_name = img_name.split("/")[-1]
            is_right = '错误'
            add_log(f'图片{img_name}预测类别为{predict_class}，真实类别为{real_class},预测{is_right}',
                    txt_list)
        # toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
        # pic = toPIL(img[0])
        # plt.imshow(pic)
        # plt.show()
        test_num += 1
    acc = right_num / test_num * 100
    add_log(f'测试总数量为{test_num}，错误数量为{test_num - right_num}', txt_list)
    add_log(f'总预测正确率为{acc}%', txt_list)
    return acc


if __name__ == '__main__':
    txt_list = []
    model_path = 'model-googlenet'
    # model_path = 'model-googlenet-2023-10-08-21h 57m'
    t_img(txt_list, model_path)
