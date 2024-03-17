import argparse
import os
import shutil
import traceback
from ultralytics import YOLO
import cv2
from yolo_config import Y_cfg
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


def detect_out(eq, sq, q, model, Y_cfg, lock, n, t_s, v):
    while True:
        if not sq.empty():
            break
    while True:
        if eq.empty() or not q.empty():
            path_list = q.get()
            # print(path_list)
            i_path, o_path = path_list
            results = model(i_path, conf=Y_cfg['detect_conf'], workers=Y_cfg['workers'], verbose=False)
            count = 1
            img = cv2.imread(i_path)
            boxes = results[0].boxes  # Boxes object for bbox outputs
            for j in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes.xyxy[j]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                save_img = img[y1:y2, x1:x2]
                cv2.imwrite(f'{o_path}_{str(count)}.jpg', save_img)
                count += 1
            lock.acquire()
            bar(v.value + 1, n, t_s)
            v.value += 1
            lock.release()
        else:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='choose model path')
    parser.add_argument('--model', type=str, default='runs/detection/result/weights/best_openvino_model/',
                        help='trained model path')
    parser.add_argument('--pn', type=int, default=2, help='process_number')
    args = parser.parse_args()
    model = YOLO(args.model)
    in_path = Y_cfg['detect_in']
    out_path = Y_cfg['detect_out']
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
    n = len(path_list)
    value = Value('i', 0)
    q = Queue(10)
    sq = Queue(1)
    eq = Queue(1)
    t_s = time.perf_counter()
    p_l = []
    for i in range(args.pn):
        p = Process(target=detect_out, args=(eq, sq, q, model, Y_cfg, lock, n, t_s, value))
        p.start()
        p_l.append(p)
    print(p_l)
    sq.put(1)
    while path_list:
        if not q.full():
            p_list = path_list.pop()
            q.put(p_list)
    eq.put(1)
    for p in p_l:
        p.join()

# detect_out(Y_cfg['detect_in'], Y_cfg['detect_out'], model, Y_cfg)
