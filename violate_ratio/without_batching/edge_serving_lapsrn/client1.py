
import grpc
import inference_pb2 as data_pb2
import inference_pb2_grpc as data_pb2_grpc
from multiprocessing import Process
from threading import Thread
import torch
import io
import argparse
import numpy as np
from time import time, time_ns
from time import sleep
#string = base64.b64encode(buffer)
from random import random
import csv
import pandas as pd

from dnn_model.lapsrn import Net
def profile_model():
    model = Net()
    model.set_profile(True)
    model.set_input([0, 39])
    model.set_profile(False)
    model_input = model.get_input()
    del model
    return model_input

model_input = profile_model()
start1 = 0
end1 = 47
start2 = 39
end2 = 47

data1 = model_input[0]
data2 = model_input[1]

parser = argparse.ArgumentParser()
parser.add_argument('--bs', type=int)
parser.add_argument('--load', type=str, default='high')
parser.add_argument('--worker', type=int, default=0)
args = parser.parse_args()

duration = np.zeros(args.bs)
start_time = np.zeros(args.bs, dtype=np.float128)

_HOST = '127.0.0.1'
_PORT = '808{}'.format(args.worker)

def guarantee_ratio(duration):
    len_duration = duration.shape[0]
    l2 = np.where(duration > 0.2 * 1.02)[0].shape[0]
    l10 = np.where(duration > 0.2 * 1.1)[0].shape[0]
    l20 = np.where(duration > 0.2 * 1.2)[0].shape[0]
    l30 = np.where(duration > 0.2 * 1.3)[0].shape[0]
    l2 = (l2 - l10)/len_duration * 100
    l10 = (l10 - l20)/len_duration * 100
    l20 = (l20 - l30)/len_duration * 100
    l30 = l30/len_duration * 100
    return l2, l10, l20, l30

def run(query_id, query_type):
    qtime = 0
    if(query_type < 0.5):
        start = start2
        end = end2
        #sleep(0.03)
        tensor = data2
        #qtime = 0.05 + qtime + 0.043/q
        qtime = qtime + np.random.uniform(0.0503, 0.0509)
        query = 1
        
    else:
        start = start1
        end = end1
        tensor = data1
        qtime = qtime + np.random.uniform(0, 0.0002)
        query = 0
    
    
    conn = grpc.insecure_channel(_HOST + ':' + _PORT)  
    client = data_pb2_grpc.FormatDataStub(channel=conn)  
    #start1 = time() 
    response = client.DoFormat(data_pb2.actionrequest(text=tensor, start=start, end=end))
    start_time_id = time()
    #end1 = time()
    #print(query, ' ', float(response.text) + time)
    duration[query_id] = float(response.text) + qtime
    start_time[query_id] = start_time_id
    #print(query, ' ', float(response.text) + qtime)
    #print(response.text)
 
if __name__ == '__main__':
    
    plist = []
    query_list = []

    for i in range(0, args.bs):
        query_list.append(random())
    for i in range(0, args.bs):
        p = Thread(target=run, args=(i, query_list[i]))
        plist.append(p)
    for item in plist:
        #start = time()
        item.start()
        # ICE
        if(args.load == 'high'):
            sleep(0.00005)
        elif(args.load == 'medium'):
            sleep(0.0015)
        elif(args.load == 'low'):
            sleep(0.0045)
        else:
            sleep(0.0009)

    for item in plist:
        item.join()
    
    duration = np.sort(duration)
    l2, l10, l20, l30 = guarantee_ratio(duration)
    print(l2, l10, l20, l30)