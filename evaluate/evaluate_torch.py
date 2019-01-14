# -*- coding: utf-8 -*-
"""
Created on 2018-12-04 10:23:06

@author: AN Zhulin
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import neat.genome
import math
import numpy as np


class Net(nn.Module):

    def __init__(self, config, genome: neat.genome.DefaultGenome):
        nn.Module.__init__(self)
        # 根据genome的连接、节点数，设置边的权重和节点的权重
        self.old_connections = genome.connections
        #print(len(self.old_connections))
        self.old_layer = genome.layer
        self.old_nodes = genome.nodes
        self.num_cnn_layer = config.genome_config.num_cnn_layer
        self.num_layer = config.genome_config.num_layer
        self.num_inputs = config.genome_config.num_inputs
        self.num_outputs = config.genome_config.num_outputs
        self.num_first_fc_layer_node = config.genome_config.num_first_fc_layer_node
        #print(len(self.old_nodes))

        self.out_channels_list = list()

        self.set_layers(genome)
        self.set_parameters(genome)



    def forward(self, x):
        l = list(self.children())
        #print(len(l))
        dropout = nn.Dropout(p=0.25)

        out_channels_list_counter = 0
        for i in range(self.num_cnn_layer):
            x = l[2 * i](x)
            x = nn.BatchNorm2d(num_features=self.out_channels_list[out_channels_list_counter], affine=True).cuda()(x)
            out_channels_list_counter += 1
            x = F.relu(x)

            x = l[2 * i + 1](x)
            x = nn.BatchNorm2d(num_features=self.out_channels_list[out_channels_list_counter], affine=True).cuda()(x)
            out_channels_list_counter += 1
            x = F.relu(x)
            if (i % 2 == 1):
                x = F.max_pool2d(x, 2)

        x = x.view(-1, self.num_flat_features)

        for i in range(self.num_cnn_layer, self.num_layer - 1):
            x = dropout(x)
            x = F.relu(l[i + self.num_cnn_layer](x))

        x = dropout(x)
        x = l[-1](x)

        return (x)

    def forward_without_dropout(self, x):
        l = list(self.children())
        #print(len(l))

        out_channels_list_counter = 0
        for i in range(self.num_cnn_layer):
            x = l[2 * i](x)
            x = nn.BatchNorm2d(num_features=self.out_channels_list[out_channels_list_counter], affine=True).cuda()(x)
            out_channels_list_counter += 1
            x = F.relu(x)

            x = l[2 * i + 1](x)
            x = nn.BatchNorm2d(num_features=self.out_channels_list[out_channels_list_counter], affine=True).cuda()(x)
            out_channels_list_counter += 1
            #x = F.batch_norm(x)
            x = F.relu(x)
            if (i % 2 == 1):
                x = F.max_pool2d(x, 2)

        x = x.view(-1, self.num_flat_features)

        for i in range(self.num_cnn_layer, self.num_layer - 1):
            x = F.relu(l[i + self.num_cnn_layer](x))

        x = l[-1](x)

        return (x)

    #def activate(self, x):
    #    return self.forward(x)

    def set_layers(self, genome: neat.genome.DefaultGenome):
        #calculate channel for every cnn layers
        cnn_channel = list()
        cnn_channel.append(self.num_inputs)
        for i in range(self.num_cnn_layer):
            cnn_channel.append(len(genome.layer[i][1]))

        #add cnn layers
        layer_id = 0
        for i in range(self.num_cnn_layer):
            self.add_module(str(layer_id), nn.Conv2d(in_channels = cnn_channel[i], out_channels = cnn_channel[i+1], kernel_size = 1, padding = 0))
            layer_id += 1
            self.out_channels_list.append(cnn_channel[i+1])

            self.add_module(str(layer_id), nn.Conv2d(in_channels = cnn_channel[i+1], out_channels = cnn_channel[i+1], kernel_size = 3, padding = 1, groups = cnn_channel[i+1]))
            layer_id += 1
            self.out_channels_list.append(cnn_channel[i+1])

        #calculate channel for every cnn layers
        fc_channel = list()
        self.num_flat_features = len(genome.layer[self.num_cnn_layer - 1][1]) * \
                                 len(genome.layer[self.num_cnn_layer][1])
        fc_channel.append(self.num_flat_features)
        for i in range(self.num_cnn_layer, self.num_layer):
            fc_channel.append(len(genome.layer[i][1]))

        #add fc layer
        for i in range(self.num_layer - self.num_cnn_layer):
            self.add_module(str(layer_id), nn.Linear(fc_channel[i], fc_channel[i+1]))
            layer_id += 1

        #set all weight and bias to zero
        for module in self.children():
            module.weight.data.fill_(0.0)
            module.bias.data.fill_(0.0)

    def set_parameters(self, genome: neat.genome.DefaultGenome):

        layer = list(self.children())#make sure change layer can affect parameters in cnn

        nodes = {}


        #add the input node to nodes dict
        order = 0
        for i in range(-self.num_inputs, 0):
            position = [-1, order]   #-1 means input node
            nodes.update({i: position})
            order += 1

        #add every layers to nodes dict
        for i in range(self.num_layer):
            l = list(genome.layer[i][1])
            l.sort()
            order = 0
            for j in range(len(l)):
                #add node (id, [layer, order in layer]
                position = [i, j]
                nodes.update({l[j]: position})

                #add conv kernel and bias to pytorch module
                if i < self.num_cnn_layer:
                    a = np.array(self.old_nodes[l[j]].kernal)
                    layer[i * 2 + 1].weight.data[j] = torch.FloatTensor(a.reshape(3, 3))
                    b = self.old_nodes[l[j]].bias
                    layer[i * 2 + 1].bias.data[j] = torch.FloatTensor([b])
                else:
                    b = self.old_nodes[l[j]].bias
                    layer[i + self.num_cnn_layer].bias.data[j] = torch.FloatTensor([b])


        for in_node, out_node in genome.connections:

            c = nodes[out_node][0] #layer number
            if c < self.num_cnn_layer: #cnn layer
                layer_num = 2 *c
                weight_tensor_num = nodes[out_node][1]
                weight_num = nodes[in_node][1]
                (layer[layer_num].weight.data[weight_tensor_num])[weight_num] = \
                    torch.FloatTensor([genome.connections[(in_node, out_node)].weight])
            elif c != self.num_cnn_layer:
                layer_num = self.num_cnn_layer + c
                weight_tensor_num = nodes[out_node][1]
                weight_num = nodes[in_node][1]
                (layer[layer_num].weight.data[weight_tensor_num])[weight_num] = \
                    torch.FloatTensor([genome.connections[(in_node, out_node)].weight])
            else:
                #print(len(layer[6].weight[0]))
                layer_num = self.num_cnn_layer + c
                weight_tensor_num = nodes[out_node][1]
                weight_num = nodes[in_node][1] * self.num_first_fc_layer_node + nodes[out_node][1]
                (layer[layer_num].weight.data[weight_tensor_num])[weight_num] = \
                    torch.FloatTensor([genome.connections[(in_node, out_node)].weight])

    def write_back_parameters(self, genome: neat.genome.DefaultGenome):

        layer = list(self.children())#make sure change layer can affect parameters in cnn

        nodes = {}

        #add the input node to nodes dict
        order = 0
        for i in range(-self.num_inputs, 0):
            position = [-1, order]   #-1 means input node
            nodes.update({i: position})
            order += 1

        #TODO: check if it is correct
        #add every layers to nodes dict
        for i in range(self.num_layer):
            l = list(genome.layer[i][1])

            for j in range(len(l)):
                # add node (id, [layer, order in layer]
                position = [i, j]
                nodes.update({l[j]: position})
                
                #write back conv kernel and bias
                if i < self.num_cnn_layer:
                    a = np.array(layer[i * 2 + 1].weight.data[j].cpu())
                    genome.nodes[l[j]].kernal = a.reshape(9)
                    genome.nodes[l[j]].bias = layer[i * 2 + 1].bias.data[j].item()
                else:
                    genome.nodes[l[j]].bias = layer[i + self.num_cnn_layer].bias.data[j].item()

        #TODO: add write back
        for in_node, out_node in genome.connections:

            c = nodes[out_node][0] #layer number
            if c < self.num_cnn_layer: #cnn layer
                layer_num = 2 *c
                weight_tensor_num = nodes[out_node][1]
                weight_num = nodes[in_node][1]
                genome.connections[(in_node, out_node)].weight = \
                    (layer[layer_num].weight.data[weight_tensor_num])[weight_num].item()
            elif c != self.num_cnn_layer:
                layer_num = self.num_cnn_layer + c
                weight_tensor_num = nodes[out_node][1]
                weight_num = nodes[in_node][1]
                genome.connections[(in_node, out_node)].weight = \
                    (layer[layer_num].weight.data[weight_tensor_num])[weight_num].item()
            else:
                #print(len(layer[6].weight[0]))
                layer_num = self.num_cnn_layer + c
                weight_tensor_num = nodes[out_node][1]
                weight_num = nodes[in_node][1] * self.num_first_fc_layer_node + nodes[out_node][1]
                genome.connections[(in_node, out_node)].weight = \
                    (layer[layer_num].weight.data[weight_tensor_num])[weight_num].item()
