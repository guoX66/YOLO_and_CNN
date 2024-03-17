import argparse
import os
import shutil
import traceback
import torch
from ultralytics import YOLO
import cv2
from _utils.y_config import Y_cfg
import time
from multiprocessing import Process, Queue, Lock, Value


def bar(i, t, start):
    l = 50
    f_p = i / t
    n_p = (t - i) / t
    finsh = "▓" * int(f_p * l)
    need_do = "-" * int(n_p * l)
    progress = f_p * 100
    dur = time.perf_counter() - start
    print("\r推理进度:{:^3.2f}%[{}->{}] 用时:{:.2f}s".format(progress, finsh, need_do, dur), end="")


def detect_out(eq, sq, q, model, Y_cfg, lock, n, t_s, v, conf):
    while True:
        if not sq.empty():
            break
    while True:
        if eq.empty() or not q.empty():
            path_list = q.get()
            # print(path_list)
            i_path = [i[0] for i in path_list]
            o_path = [i[1] for i in path_list]

            results = model(i_path, imgsz=Y_cfg['imgsz'], device=Y_cfg['device'], conf=conf, workers=Y_cfg['workers'],
                            verbose=False)
            for k in range(len(results)):
                count = 1
                img = cv2.imread(i_path[k])
                result = results[k]
                boxes = result.boxes  # Boxes object for bbox outputs
                for j in range(boxes.shape[0]):
                    x1, y1, x2, y2 = boxes.xyxy[j]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    save_img = img[y1:y2, x1:x2]
                    cv2.imwrite(f'{o_path[k]}_{str(count)}.jpg', save_img)
                    count += 1
            lock.acquire()
            bar(v.value + 1, n, t_s)
            v.value += 1
            lock.release()
        else:
            break


if __name__ == '__main__':
    in_path = 'data/c_data'
    out_path = 'CNN_IC/data/static'
    model_path = Y_cfg['out_file']

    parser = argparse.ArgumentParser(description='choose model path')
    parser.add_argument('--model_type', type=str, default='pt',
                        help='trained model path')
    parser.add_argument('--model', type=str, default=f'runs/detect/{model_path}/weights/best.pt',
                        help='trained model path')

    parser.add_argument('--pn', type=int, default=2, help='process_number')
    parser.add_argument('--detect_piece', type=int, default=100)
    parser.add_argument('--conf', type=float, default=0.5)

    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')
    model_path = args.model
    model_path = model_path.replace('pt', args.model_type)
    model = YOLO(model_path)
    try:
        shutil.rmtree(out_path)
    except:
        pass
    os.mkdir(out_path)
    i_path = []
    o_path = []
    path_list = []
    path_count = 0
    for i in os.walk(in_path):
        if path_count == 0:
            for label in i[1]:
                label_path = os.path.join(out_path, label)
                os.mkdir(label_path)
        else:
            file_path = i[2]
            for j in file_path:
                in_path_1 = os.path.join(i[0], j)
                out_path1 = in_path_1.replace(in_path, out_path)
                dirStr, ext = os.path.splitext(out_path1)
                path_list.append((in_path_1, dirStr))

        path_count += 1

    lock = Lock()
    piece = args.detect_piece
    b = [path_list[i:i + piece] for i in range(0, len(path_list), piece)]
    n = len(b)
    value = Value('i', 0)
    q = Queue(10)
    sq = Queue(1)
    eq = Queue(1)
    t_s = time.perf_counter()
    p_l = []
    for i in range(args.pn):
        p = Process(target=detect_out, args=(eq, sq, q, model, Y_cfg, lock, n, t_s, value, args.conf))
        p.start()
        p_l.append(p)
    sq.put(1)
    print(p_l)
    while b:
        if not q.full():
            p_list = b.pop()
            q.put(p_list)
    eq.put(1)
    for p in p_l:
        p.join()

# detect_out(Y_cfg['detect_in'], Y_cfg['detect_out'], model, Y_cfg)
