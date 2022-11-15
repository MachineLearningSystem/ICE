import csv
import math
import numpy as np
import copy
import argparse
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--dnn', type=str, default='lapsrn')
parser.add_argument('--net', type=float, default=5)
parser.add_argument('--mobile', type=str, default='kirin')
args = parser.parse_args()

dnn = args.dnn
mobile = args.mobile
folder = 'dag/'




start = time()

file = dnn + '_data.csv'
file = folder + file
f = open(file ,'r')
csv_reader = csv.reader(f)
DAG_profile_cloud = []
DAG_profile_edge = []
DAG_profile_trans = []
for i, line in enumerate(csv_reader):
    if (i > 0):
        DAG_profile_cloud.append(float(line[6]))
        if(args.mobile == 'kirin'):
            DAG_profile_edge.append(float(line[1]))
        elif(args.mobile == 'mi'):
            DAG_profile_edge.append(float(line[2]))
        elif(args.mobile == 'pi'):
            DAG_profile_edge.append(float(line[3]))
        if(args.net == 5):
            DAG_profile_trans.append(float(line[4]))
        elif(args.net == 4):
            DAG_profile_trans.append(float(line[5]))

DAG_profile_cloud = np.array(DAG_profile_cloud)
DAG_profile_edge = np.array(DAG_profile_edge)
DAG_profile_trans = np.array(DAG_profile_trans)

length = len(DAG_profile_cloud)
time = 100000000
cut = -1
for slice in range(0, length):
    t_cloud = np.sum(DAG_profile_cloud[slice:length])
    t_edge = np.sum(DAG_profile_edge[0:slice])
    t_tran = DAG_profile_trans[slice]
    t_e2e = t_cloud + t_edge + t_tran + DAG_profile_trans[-1]
    if(t_e2e < time):
        cut = slice
        time = t_e2e
trans_start = DAG_profile_trans[0]
cloud_d = np.sum(DAG_profile_cloud)
edge_d = np.sum(DAG_profile_edge)
device = 1
if(edge_d < cloud_d + trans_start):
    device = 0
print(1-cut/length, device)


