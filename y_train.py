import shutil
from ultralytics import YOLO
from C.utils import divide_dataset
from Y.y_config import Y_cfg, train_data, curpath, make_yaml

if __name__ == '__main__':
    make_yaml(curpath, train_data)
    model = YOLO(Y_cfg['model'])
    if Y_cfg['is_divide']:
        t, v = int(Y_cfg['train_divide'][0]), int(Y_cfg['train_divide'][2])
        divide_dataset(Y_cfg['divide_in'], Y_cfg['divide_out'], t, v)
    # 训练模型
    output_file = Y_cfg['out_file']
    try:
        shutil.rmtree(f'runs/detect/{output_file}')
    except:
        pass
    results = model.train(name=output_file,
                          data=Y_cfg['train_data'],
                          epochs=Y_cfg['epochs'],
                          imgsz=Y_cfg['imgsz'],
                          workers=Y_cfg['workers'],
                          batch=Y_cfg['batch'],
                          device=Y_cfg['device'],
                          amp=Y_cfg['AMP']
                          )
