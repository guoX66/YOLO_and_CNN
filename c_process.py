# coding:utf-8
from train import train_process
from configs import TrainImg
from utils import divide_test
import multiprocessing

Train = TrainImg()
if Train.is_divide:
    divide_test(Train.data_path, Train.train_path, Train.t_divide_present)
p_list=[]
net_list=[('resnet18','DenseNet201'),('resnet101','DenseNet121'),('googlenet', 'DenseNet161'), ('resnet34', 'resnet50')]

for i,j in net_list:
      p1 = multiprocessing.Process(target=train_process, args=(i, ))
      p2 = multiprocessing.Process(target=train_process, args=(j, ))
      p1.start()
      p2.start()
      p1.join()
      p2.join()

    

    
    #try:
     #   train_process(i)
    #except Exception as ex:
      #  print(ex)
       # raise ex
