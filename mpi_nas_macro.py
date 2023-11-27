import os
import json
import openai
import numpy as np
from decimal import Decimal
import argparse
from subprocess import Popen


import torch
import torch.nn as nn
import torch.optim as optim
from mpi4py import MPI
import io
import mpi_chexpert
import gc  # Import garbage collection module

from mpi_chexpert import CModel


#### mpiexec -np 3 python mpi_nas_macro.py --openai_key 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 --openai_organization org-pYjfh3VvqELRH9LJaGfxaxf8 --train --n_epochs 1 --plot_roc --batch_size 24 --model NetWork 


### Error cuda device

# sudo rmmod nvidia_uvm
# sudo modprobe nvidia_uvm

parser = argparse.ArgumentParser()
# action
parser.add_argument('--load_config', type=str, help='Path to config.json file to load args from.')
parser.add_argument('--train', action='store_true', help='Train model.')
parser.add_argument('--evaluate_single_model', action='store_true', help='Evaluate a single model.')
parser.add_argument('--evaluate_ensemble', action='store_true', help='Evaluate an ensemble (given a checkpoints tracker of saved model checkpoints).')
parser.add_argument('--visualize', action='store_true', help='Visualize Grad-CAM.')
parser.add_argument('--plot_roc', action='store_true', help='Filename for metrics json file to plot ROC.')
parser.add_argument('--seed', type=int, default=0, help='Random seed to use.')
#parser.add_argument('--cuda', type=int, help='Which cuda device to use.')
# pathsbatch_
parser.add_argument('--data_path', 
default='', help='Location of train/valid datasets directory or path to test csv file.')
parser.add_argument('--output_dir', help='Path to experiment output, config, checkpoints, etc.')
parser.add_argument('--restore', type=str, help='Path to a single model checkpoint to restore or folder of checkpoints to ensemble.')
# model architecture
parser.add_argument('--model', default='densenet121', help='What model architecture to use. (densenet121, resnet152, efficientnet-b[0-7])')
# data params
parser.add_argument('--mini_data', type=int, help='Truncate dataset to this number of examples.')
parser.add_argument('--resize', type=int, help='Size of minimum edge to which to resize images.')
# training params
parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained model and normalize data mean and std.')
parser.add_argument('--batch_size', type=int, default=16, help='Dataloaders batch size.')
parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--lr_warmup_steps', type=float, default=0, help='Linear warmup of the learning rate for lr_warmup_steps number of steps.')
parser.add_argument('--lr_decay_factor', type=float, default=0.97, help='Decay factor if exponential learning rate decay scheduler.')
parser.add_argument('--step', type=int, default=0, help='Current step of training (number of minibatches processed).')
parser.add_argument('--log_interval', type=int, default=50, help='Interval of num batches to show loss statistics.')
parser.add_argument('--eval_interval', type=int, default=300, help='Interval of num epochs to evaluate, checkpoint, and save samples.')
#parser.add_argument('--arch', type=str, default='11111111', help='which architecture to use')




parser.add_argument('--openai_key', type=str, required=True)
parser.add_argument('--openai_organization', type=str, required=True)
args = parser.parse_args()

openai.api_key = args.openai_key
openai.organization = args.openai_organization


benchmark_file = open('benchmark/nas-bench-macro_cifar10.json')
data = json.load(benchmark_file)
keys = list(data.keys())
rank = np.array([data[k]['mean_acc'] for k in keys]).argsort().argsort()
for k, r in zip(keys, rank):
    data[k]['rank'] = (3 ** 8) - r

system_content = "You are an expert in the field of neural architecture search."

# user_input = '''Your task is to assist me in selecting the best operations for a given models architectures, which includes some undefined layers and available operations. The models will be trained and tested on ChexPert dataset, and your objective will be to maximize the model's performance on Chexpert dataset.

# We define the 3 available operations as the following:
# 0: Identity(in_channels, out_channels, stride)
# 1: InvertedResidual(in_channels, out_channels, stride expansion=3, kernel_size=3)
# 2: InvertedResidual(in_channels, out_channels, stride expansion=6, kernel_size=5)

# The implementation of the Identity is as follows:
# class Identity(nn.Module):
#     def __init__(self, in_channels, out_channels, stride):
#         super(Identity, self).__init__()
#         if stride != 1 or in_channels != out_channels:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels),
#             )
#         else:
#             self.downsample = None

#     def forward(self, x):
#         if self.downsample is not None:
#             x = self.downsample(x)
#         return x

# The implementation of the InvertedResidual is as follows:
# class InvertedResidual(nn.Module):
#     def __init__(self, in_channels, out_channels, stride, expansion, kernel_size):
#         super(InvertedResidual, self).__init__()
#         hidden_dim = in_channels * expansion
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=hidden_dim, bias=False),
#             nn.BatchNorm2d(hidden_dim),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(out_channels),
#         )
#         self.use_shortcut = in_channels == out_channels and stride == 1

#     def forward(self, x):
#         if self.use_shortcut:
#             return self.conv(x) + x
#         return self.conv(x)
        

# The model architecture will be defined as the following.
# {
#     layer1:  {defined: True,  operation: nn.Conv2d(in_channels=3,  out_channels=32, kernel_size=3, padding=1, bias=False)},
#     layer2:  {defined: False, downsample: True , in_channels: 32,  out_channels: 64 , stride: 2},
#     layer3:  {defined: False, downsample: False, in_channels: 64,  out_channels: 64 , stride: 1},
#     layer4:  {defined: False, downsample: True , in_channels: 64,  out_channels: 128, stride: 2},
#     layer5:  {defined: False, downsample: False, in_channels: 128, out_channels: 128, stride: 1},
#     layer6:  {defined: False, downsample: False, in_channels: 128, out_channels: 128, stride: 1},
#     layer7:  {defined: False, downsample: True , in_channels: 128, out_channels: 256, stride: 2},
#     layer8:  {defined: False, downsample: False, in_channels: 256, out_channels: 256, stride: 1},
#     layer9:  {defined: False, downsample: False, in_channels: 256, out_channels: 256, stride: 1},
#     layer10: {defined: True,  operation: nn.Conv2d(in_channels=256, out_channels=1280, kernel_size=1, bias=False, stride=1)},
#     layer11: {defined: True,  operation: nn.AdaptiveAvgPool2d(output_size=1)},
#     layer12: {defined: True,  operation: nn.Linear(in_features=1280, out_features=10)},
# }

# The currently undefined layers are layer2 - layer9, and the in_channels and out_channels have already been defined for each layer. To maximize the model's performance on Chexpert dataset, please provide me with your suggested operation for the undefined layers only. 

# Your response for each model should be an operation ID list for the undefined layers. For example:
# [1, 2, ..., 0] means we use operation 1 for layer2, operation 2 for layer3, ..., operation 0 for layer9.
# '''

number_gpu=2;
string_number_gpu = "two";

user_input = '''Your task is to assist me in selecting the best operations for a given ''' + string_number_gpu + ''' models architectures, which includes eight undefined layers and available operations. The ''' + string_number_gpu + ''' models will be trained and tested on ChexPert dataset, and your objective will be to maximize the model's performance on Chexpert dataset.

We define the 3 available operations as the following:
0: Identity(in_channels, out_channels, stride)
1: InvertedResidual(in_channels, out_channels, stride expansion=3, kernel_size=3)
2: InvertedResidual(in_channels, out_channels, stride expansion=6, kernel_size=5)

The implementation of the Identity is as follows:
class Identity(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Identity, self).__init__()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return x

The implementation of the InvertedResidual is as follows:
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion, kernel_size):
        super(InvertedResidual, self).__init__()
        hidden_dim = in_channels * expansion
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.use_shortcut = in_channels == out_channels and stride == 1

    def forward(self, x):
        if self.use_shortcut:
            return self.conv(x) + x
        return self.conv(x)
        

The ''' + string_number_gpu + ''' models architectures will be defined as the following.
{
    layer1:  {defined: True,  operation: nn.Conv2d(in_channels=3,  out_channels=32, kernel_size=3, padding=1, bias=False)},
    layer2:  {defined: False, downsample: True , in_channels: 32,  out_channels: 64 , stride: 2},
    layer3:  {defined: False, downsample: False, in_channels: 64,  out_channels: 64 , stride: 1},
    layer4:  {defined: False, downsample: True , in_channels: 64,  out_channels: 128, stride: 2},
    layer5:  {defined: False, downsample: False, in_channels: 128, out_channels: 128, stride: 1},
    layer6:  {defined: False, downsample: False, in_channels: 128, out_channels: 128, stride: 1},
    layer7:  {defined: False, downsample: True , in_channels: 128, out_channels: 256, stride: 2},
    layer8:  {defined: False, downsample: False, in_channels: 256, out_channels: 256, stride: 1},
    layer9:  {defined: False, downsample: False, in_channels: 256, out_channels: 256, stride: 1},
    layer10: {defined: True,  operation: nn.Conv2d(in_channels=256, out_channels=1280, kernel_size=1, bias=False, stride=1)},
    layer11: {defined: True,  operation: nn.AdaptiveAvgPool2d(output_size=1)},
    layer12: {defined: True,  operation: nn.Linear(in_features=1280, out_features=10)},
}

The currently eight undefined layers are layer2 - layer9, that is eight layers and the in_channels and out_channels have already been defined for each layer. To maximize the model's performance on Chexpert dataset, please provide me with your suggested operation for the undefined layers only. 

Your response for each model should be an operation ID list for the eight undefined layers. For example:
[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0] means we use operation 0 for layer2, operation 0 for layer3, operation 0 for layer4, operation 0 for layer5,operation 0 for layer6, operation 0 for layer7, operation 0 for layer8 and operation 0 for layer9 of the first model.
'''



experiments_prompt = lambda arch_list, acc_list : '''Here are some experimental results that you can use as a reference:
{}
Please suggest a better operation ID list that can improve the model's performance on Chexpert dataset beyond the experimental results provided above.
'''.format(''.join(['{} gives an accuracy of {:.2f}%\n'.format(arch, acc) for arch, acc in zip(arch_list, acc_list)]))

suffix = '''Please do not include anything other than the operations ID list of list in your response for the ''' + string_number_gpu + ''' models.'''



arch_list = []
acc_list = []
previuos_calculated=False


# Check if the file exists
if os.path.exists("architectures.json"):
    previuos_calculated = True
    with open("architectures.json", 'r') as file:
        arch_list= json.load(file)

    if os.path.exists("accuaricies.json"):
        with open("accuaricies.json", 'r') as file:
            acc_list= json.load(file)
        messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_input + experiments_prompt(arch_list, acc_list) + suffix},
            ]
    else:
        previuos_calculated = False


print("MESSAGES = " ,messages)


performance_history = []
messages_history = []

if not os.path.exists('history'):
    os.makedirs('history')


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

    
for iteration in range(10):
    if rank == 0:
        print("messages : ", iteration);
        print(messages);
        print("/n/n");
        print("/n/n");
        res = openai.ChatCompletion.create(model='gpt-4-0314', messages=messages, temperature=0, n=1)['choices'][0]['message']
        str_lists =  res['content'] 
        print(str_lists)
        if(str_lists.count('\n')>0):
            str_lists = str_lists.replace('],',']');
        else:
            str_lists = str_lists.replace('], ',']\n');
       
        str_lists = str_lists.strip().split('\n')
        
        # Convert each string list to a list of integers
        list_of_lists = [list(map(int, s.strip('[]').split(', '))) for s in str_lists]

        messages.append(res)
        messages_history.append(messages)

        print("res[", iteration, "][0] = ", list_of_lists[0], "\n")
        print("res[", iteration, "][1] = ", list_of_lists[1], "\n")
        
        
        # Master node
        for i in range(1, size):
            arch_list.append(list_of_lists[i-1])
            model_structure_str = ''.join(str(opid) for opid in list_of_lists[i-1])
            comm.send(model_structure_str, dest=i)

        # Gather results from worker nodes
        accuracies = []
        for i in range(1, size):
            accuracy = comm.recv(source=i)
            acc_list.append(accuracy)
            accuracies.append(accuracy)
        print("Accuracies from worker nodes:", accuracies)

        with open("architectures.json", 'w') as file:
            json.dump(arch_list, file, indent=4)
        with open("accuaricies.json", 'w') as file:
            json.dump(acc_list, file, indent=4)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_input + experiments_prompt(arch_list, acc_list) + suffix},
        ]

    else:
        # Worker nodes
        # Receive the model
        data = comm.recv(source=0)
       # buffer = io.BytesIO(data)
        print("Process ", rank , " received data: " , data)
        # Train and evaluate the model
        accuracy = CModel.chexpert(args, data, rank)

         # After training and evaluation, if 'data' is large and no longer needed
        del data  # Explicitly delete it

        # Send the accuracy back to the master node
        comm.send(accuracy, dest=0)

        # Manually trigger garbage collection
        gc.collect()



        
