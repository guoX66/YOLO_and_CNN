import json
import os
import platform
import torch
import argparse
import yaml

train_data = 'Y/mydata.yaml'
current_dir = os.path.dirname(os.path.realpath(__file__))
curpath = os.path.dirname(current_dir)

os_name = str(platform.system())


parser = argparse.ArgumentParser()
parser.add_argument('--is_divide', type=bool, default=False)
parser.add_argument('--model', type=str, default="Y/yaml/yolov8n.yaml")
parser.add_argument('--divide_rate', type=str, default="8:2")
parser.add_argument('--out_file', default=None)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--imgsz', type=int, default=640)
parser.add_argument('--conf', type=float, default=0.3)
parser.add_argument('--amp', type=bool, default=False)
args = parser.parse_args()

model = args.model
if os_name == 'Windows':
    num_workers = 0
    dirStr, ext = os.path.splitext(model)
else:
    num_workers = 32
    dirStr, ext = os.path.splitext(model)

file = dirStr.split("/")[-1]
if torch.cuda.is_available():
    gpu_num = torch.cuda.device_count()
    device = [i for i in range(gpu_num)]
else:
    device = torch.device('cpu')

Y_cfg = {
    'model': model,
    'xml_path': f'{curpath}/y_data/xml_data',
    'is_divide': args.is_divide,
    'divide_in': f'{curpath}/y_data/i_datasets',
    'divide_out': f'{curpath}/y_data/datasets',
    'train_divide': args.divide_rate,
    'train_data': train_data,
    'out_file': args.out_file if args.out_file else file,
    'device': device,
    'workers': num_workers,
    'epochs': args.epochs,
    'imgsz': args.imgsz,
    'batch': args.batch,
    'conf': args.conf,
    'AMP': args.amp,

}


def make_yaml(curpath, train_data):
    with open("./log/y_class.json", 'r', encoding='UTF-8') as f:
        class_dict = json.load(f)
    class_dict = {int(i): class_dict[i] for i in class_dict.keys()}

    desired_caps = {

        'path': f'{curpath}/y_data/datasets',  # dataset root dir
        'train': 'train/images',
        'val': 'val/images',
        # Classes
        'names': class_dict

    }
    with open(train_data, "w", encoding="utf-8") as f:
        yaml.dump(desired_caps, f)
