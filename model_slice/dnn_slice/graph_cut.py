import csv
import math
import numpy as np
import copy
import argparse
from time import time
def Deep_search(DAG, now_layer, branch_deep, DAG_Output, is_skip_branch=False):

    if(now_layer == -1):
        return DAG_Output, -1
    now_node = DAG[now_layer]
    next_layer = -1
    if(now_layer == -1):
        return DAG_Output, -1
    if((DAG_Name[now_layer] == "aten::add" or DAG_Name[now_layer] == "aten::cat") and is_skip_branch == False):
        return DAG_Output, now_layer

    if(len(now_node) == 1):
        next_node = DAG[now_layer][0]
        if(DAG_Name[now_layer] != "aten::add" and DAG_Name[now_layer] != "aten::cat"):
            DAG_Output.append(now_layer)
        DAG_Output, next_layer = Deep_search(DAG, next_node, branch_deep, DAG_Output)
    elif(len(now_node) == 0):
        DAG_Output.append(now_layer)
        DAG_Output, next_layer = Deep_search(DAG, -1, branch_deep, DAG_Output)
    elif(len(now_node) > 0): #if one layer have more than one branch this layer will be a big node
        big_node = []
        if(branch_deep > 0):
            DAG_Output.append(now_layer)
        for next_node in DAG[now_layer]:
            if(branch_deep == 0):
                branch_node = [now_layer]
                branch_node, next_layer = Deep_search(DAG, next_node, branch_deep + 1, branch_node)
                branch_node.append(next_layer)
                if(len(branch_node) > 2):
                    big_node.append(branch_node)
            else:
                DAG_Output, next_layer = Deep_search(DAG, next_node, branch_deep + 1, DAG_Output)
        next_node = next_layer
        if(branch_deep == 0):
            if(len(big_node) > 1):
                DAG_Output.append(big_node)
            else:
                for item in big_node[0]:
                    DAG_Output.append(item)
            DAG_Output, next_layer = Deep_search(DAG, next_node, branch_deep, DAG_Output, True)
        else:
            DAG_Output, next_layer = Deep_search(DAG, next_node, branch_deep, DAG_Output, True)
            DAG_Output.append(next_layer)

    return DAG_Output, next_layer
        

def build_execute_graph(DAG_Output, DAG_profile_cloud, DAG_profile_edge, DAG_trans):

    execute_graph = []
    for index, item in enumerate(DAG_Output):
        stage_graph = []
        stage_min = []
        if(isinstance(item, int)): 
            stage_graph.append(DAG_profile_cloud[item])
            stage_graph.append(DAG_profile_cloud[item] + DAG_trans[item])
            stage_graph.append(DAG_profile_edge[item] + DAG_trans[item])
            stage_graph.append(DAG_profile_edge[item])

        elif(isinstance(item, list)):
            big_node = DAG_Output[index]
            lengh = len(big_node)
            branch_pos = [0 for _ in range(lengh)]
            min_time = 10000000 
            min_pos = 0
            for top, bottom in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                min_time = 10000000
                min_pos = 0
                for num in range(int(math.pow(2, lengh))):
                    execute_time = [0, 0] 
                    trans_time = [0, 0, 0, 0] 
                    bin_num = list(bin(num))
                    while(len(bin_num) < lengh + 2):
                        bin_num.insert(2, '0')
                    for index in range(2, len(bin_num)):
                        branch_pos[index - 2] = int(bin_num[index])
                    for index, value in enumerate(branch_pos):      
                        branch = big_node[index]
                        if(value == 0):
                            execution_branch = np.array(DAG_profile_cloud)[branch]
                            execution_branch = execution_branch[1:-1]
                            trans_top = DAG_trans[branch[0]]
                            trans_bottom = DAG_trans[branch[-2]]
                            execute_time[0] += np.sum(execution_branch)
                            if(trans_time[0] == 0 and top == 1):
                                trans_time[0] += trans_top
                            if(bottom == 1):
                                trans_time[2] += trans_bottom
                        elif(value == 1):
                            execution_branch = np.array(DAG_profile_edge)[branch]
                            execution_branch = execution_branch[1:-1]
                            trans_top = DAG_trans[branch[0]]
                            trans_bottom = DAG_trans[branch[-2]]
                            execute_time[1] += np.sum(execution_branch)
                            if(trans_time[1] == 0 and top == 0):
                                trans_time[1] += trans_top
                            if(bottom == 0):
                                trans_time[3] += trans_bottom
                    profile_time = np.max([execute_time[0] + trans_time[0] + trans_time[2] ,execute_time[1] + trans_time[1] + trans_time[3]])
                    if(top == 0):
                        profile_time += DAG_profile_cloud[big_node[0][0]]
                    elif(top == 1):
                        profile_time += DAG_profile_edge[big_node[0][0]]
                    if min_time > profile_time:
                        min_time = profile_time
                        min_pos = num
                stage_graph.append(min_time)
                stage_min.append(min_pos)
            stage_graph.append(stage_min)
        execute_graph.append(stage_graph)    
    return execute_graph

def short_path(execute_path, wait_time):
    now_layer_cloud = 0
    now_layer_edge = 0
    past_layer_cloud = []
    past_layer_edge = []
    for i, item in enumerate(execute_path):
        next_cloud = [0, 0]
        next_edge = [0, 0]
        next_cloud[0] = now_layer_cloud + item[0]
        next_cloud[1] = now_layer_edge + item[2] 
        next_edge[0] = now_layer_cloud + item[1] + wait_time
        next_edge[1] = now_layer_edge + item[3]
        if(next_cloud[0] < next_cloud[1]):
            now_layer_cloud = next_cloud[0]
            tmp_cloud = copy.deepcopy(past_layer_cloud)
            tmp_cloud.append(0)

        elif(next_cloud[0] >= next_cloud[1]):
            now_layer_cloud = next_cloud[1]
            tmp_cloud = copy.deepcopy(past_layer_edge)
            tmp_cloud.append(1)

        if(next_edge[0] < next_edge[1]):
            now_layer_edge = next_edge[0]
            tmp_edge = copy.deepcopy(past_layer_cloud)
            tmp_edge.append(0)

        elif(next_edge[0] >= next_edge[1]):
            now_layer_edge = next_edge[1]
            tmp_edge = copy.deepcopy(past_layer_edge)
            tmp_edge.append(1)

        past_layer_cloud = tmp_cloud
        past_layer_edge = tmp_edge
    return now_layer_cloud, now_layer_edge, past_layer_cloud, past_layer_edge


parser = argparse.ArgumentParser()
parser.add_argument('--dnn', type=str, default='bert')
parser.add_argument('--net', type=float, default=5)
parser.add_argument('--mobile', type=str, default='kirin')
args = parser.parse_args()

DAG = []
dnn = args.dnn
mobile = args.mobile
file = dnn + '.dag'
folder = 'dag/'
file = folder + file
f = open(file, 'r')
csv_reader = csv.reader(f)
for line in csv_reader:
    for i, item in enumerate(line):
        line[i] = int(item)
    DAG.append(line)
DAG_Name = []
file = dnn + '.name'
file = folder + file
f = open(file, 'r')
csv_reader = csv.reader(f)
for line in csv_reader:
    DAG_Name.append(line[0])
DAG_Output = []


DAG_Output, _ = Deep_search(DAG, 0, 0, DAG_Output)
tmp = len(DAG)
for item in DAG_Output:
    if(isinstance(item, list)):
        if(len(item) > 2):
            tmp = len(DAG_Output)

DAG_Output_name = []
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

cloud_time = np.sum(np.array(DAG_profile_cloud)) + (DAG_profile_trans[0] + DAG_profile_trans[-1])
wait_time = np.sum(np.array(DAG_profile_cloud)) * 0
execute_graph = build_execute_graph(DAG_Output, DAG_profile_cloud, DAG_profile_edge, DAG_profile_trans)
input_trans = execute_graph[0]

input_trans[0] = 1000000
input_trans[1] = 1000000
execute_graph[0] = input_trans
now_layer_cloud, now_layer_edge, past_layer_cloud, past_layer_edge = short_path(execute_graph, wait_time)

cloud_time = np.sum(np.array(DAG_profile_cloud)) + (DAG_profile_trans[0] + DAG_profile_trans[-1])
wait_time = np.sum(np.array(DAG_profile_cloud)) * 0
execute_graph = build_execute_graph(DAG_Output, DAG_profile_cloud, DAG_profile_edge, DAG_profile_trans)
input_trans = execute_graph[0]

input_trans[0] = 1000000
input_trans[1] = 1000000
execute_graph[0] = input_trans
now_layer_cloud, now_layer_edge, past_layer_cloud, past_layer_edge = short_path(execute_graph, wait_time)

past_layer_edge = np.array(past_layer_edge)
l_len = past_layer_edge.shape[0] - 1
exe_cloud0 = -1
exe_cloud1 = -1
serve_cloud = False
while(l_len >= 0):
    if(past_layer_edge[l_len] == 0 and serve_cloud == False):
        exe_cloud1 = l_len
        serve_cloud = True
    elif(past_layer_edge[l_len] == 1 and serve_cloud == True):
        exe_cloud0 = l_len
        serve_cloud = False 
        break
    l_len = l_len - 1
ratio = (exe_cloud1 - exe_cloud0)/len(DAG_Output)
exe_cloud0 = (exe_cloud0+1)/(tmp)
exe_cloud1 = (exe_cloud1+1)/(tmp)

 
print(exe_cloud0, exe_cloud1, ratio)
for item in DAG_Output:
    if(isinstance(item, int)):
        DAG_Output_name.append(DAG_Name[item])
    else:
        tmp = []
        for i in range(len(item)):
            tmp1 = []
            for value in item[i]:
                tmp1.append(DAG_Name[value])
            tmp.append(tmp1)
        DAG_Output_name.append(tmp)