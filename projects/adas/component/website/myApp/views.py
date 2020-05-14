from django.shortcuts import render,HttpResponse,redirect
import sys
from .consumer import push
sys.path.append("/home/watrix-007/AiwSys/cyber/python/")
sys.path.append("/home/watrix-007/AiwSys/cyber/python/examples")

# sys.path.append("/home/watrix-007/AiwSys/bazel-bin/cyber/py_wrapper")
# _CYBER_INIT = importlib.import_module('_cyber_init')

# Create your views here.
import threading
from  cyber_py import cyber
from unit_test_pb2 import ChatterBenchmark
from channels.layers import get_channel_layer
flag=0 #线程开启标志,避免开启两个相同的线程

def image(request):
    global flag
    if(flag<1):
        listen = ListenThread()
        listen.start()
        flag+=1
    return render(request,'myApp/image.html')


  # 监听函数
def create_listen():
    cyber.init("web")
    node = cyber.Node("test")
    node.create_reader("channel/chatter",ChatterBenchmark, callback)
    node.spin()

# 监听回调函数
def callback(data): 
    imgs=dict()
    imgs['code']=data.content
    imgs['cout']=data.seq
    push('image',imgs)


# 监听线程
class ListenThread(threading.Thread):
    def run(self):
        create_listen()