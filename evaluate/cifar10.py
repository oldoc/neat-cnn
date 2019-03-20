import sys,os
curPath = os.path.abspath(os.path.dirname(__file__))
parentPath = os.path.split(curPath)[0]
rootPath = os.path.split(parentPath)[0]
sys.path.append(rootPath)
sys.path.append(rootPath+"/neat-cnn")

# import numpy as np
from random import random

# from __future__ import print_function
import neat

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np

import evaluate_torch
"""
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
"""

# Cutout data enhance
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
        	# (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

# Data enhance
cifar_norm_mean = (0.49139968, 0.48215827, 0.44653124)
cifar_norm_std = (0.24703233, 0.24348505, 0.26158768)


# Data enhance without cutout
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar_norm_mean, cifar_norm_std),
])
'''

# Data enhance with cutout
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar_norm_mean, cifar_norm_std),
    Cutout(n_holes=1, length=16)
])
'''

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_norm_mean, cifar_norm_std),
])

'''
# Data normalization
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
'''

torch_batch_size = 100

gpu = False

first_time = True

best_on_test_set = 0.8

#net_dict = {}

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainset.train_data = trainset.train_data[0:40000]
trainset.train_labels = trainset.train_labels[0:40000]
trainset.train_list = trainset.train_list[0:4]

trainloader = torch.utils.data.DataLoader(trainset, batch_size=torch_batch_size,
                                          shuffle=True, num_workers=2)

evaluateset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_test)
evaluateset.train_data = evaluateset.train_data[40000:50000]
evaluateset.train_labels = evaluateset.train_labels[40000:50000]
evaluateset.train_list = evaluateset.train_list[4:5]

evaluateloader = torch.utils.data.DataLoader(evaluateset, batch_size=torch_batch_size,
                                          shuffle=False, num_workers=2)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=torch_batch_size,
                                         shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 第epoch值进行计算并更新学习率
def get_adjusted_lr(epoch, T_0=150, T_mult=1, eta_max=0.1, eta_min=0.):
    i = np.log2(epoch / T_0 + 1).astype(np.int)
    T_cur = epoch - T_0 * (T_mult ** (i) - 1)
    T_i = (T_0 * T_mult ** i)
    cur_lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * T_cur / T_i))
    return cur_lr

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# evaluate the fitness
# batch_size = 0 means evaluate until reach the end of the evaluate set
def eval_fitness(net, loader, batch_size, torch_batch_size, start, gpu):

    # eval() only changes the state of some modules, e.g., dropout, but do not disable loss back-propogation.
    # By setting eval(), dropout() does not work and is temporarily removed from the chain of update.

    #switch to evaluation mode
    net.eval()

    hit_count = 0
    total = 0

    i = 0
    with torch.no_grad():
        for num, data in enumerate(loader, start):
            i += 1
            total += torch_batch_size

            # 得到输入数据
            inputs, labels = data

            # 包装数据
            if gpu:
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
            else:
                inputs, labels = inputs.to("cpu"), labels.to("cpu")
            try:

                outputs = net(inputs)
                _, predicted = outputs.max(1)
                hit_count += predicted.eq(labels).sum()

            except Exception as e:
                print(e)

            if (i == batch_size): #because i > 0 then no need to judge batch_size != 0
                break

    #switch to training mode
    net.train()

    return (hit_count.item() / total)

def eval_genomes(genomes, config):

    global gpu
    global first_time
    global best_on_test_set

    best_every_generation = list()

    if torch.cuda.is_available():
        gpu = True
        print("Running on GPU!")
    else:
        gpu = False
        print("Running on CPU!")

    """
    lrfile= open("lr.txt", "r")
    tmp = lrfile.readline().rstrip('\n')
    lr = float(tmp)
    tmp = lrfile.readline().rstrip('\n')
    delta = float(tmp)
    tmp = lrfile.readline().rstrip('\n')
    max_epoch = int(tmp)
    lrfile.close()
    """

    #genomes_id_set = set()

    j = 0
    for genome_id, genome in genomes:
        j += 1

        #setup the network, use saved net
        #if genome_id in net_dict:
        #    net = net_dict[genome_id]
        #else:

        #net = evaluate_torch.Net(config, genome)

        # load lr and epoch
        lrfile = open("lr.txt", "r")
        tmp = lrfile.readline().rstrip('\n')
        lr = float(tmp)
        tmp = lrfile.readline().rstrip('\n')
        max_epoch = int(tmp)
        lrfile.close()

        print(first_time)
        if first_time:
            net = evaluate_torch.Net(config, genome, False)
        else:
            net = evaluate_torch.Net(config, genome, True)

        if gpu:
            net.cuda()

        criterion = nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
        optimizer = optim.SGD(net.parameters(), lr, momentum=0.9, weight_decay=5e-4)

        #Evalute the fitness before trainning
        evaluate_batch_size = 0
        start = 0

        fit = eval_fitness(net, evaluateloader, evaluate_batch_size, torch_batch_size, start, gpu)

        comp = open("comp.csv", "a")
        comp.write('{0},{1:3.3f},'.format(j, fit))
        print('Before: {0}: {1:3.3f}'.format(j, fit))

        #train the network
        epoch = 0
        lr_total_reduce_times = 3  # times lr reduce to its 0.1
        lr_reduce_times = 0  # current times lr reduced

        precision_count = 0
        precision_count_max = 5
        best_precision = 0.0

        evaluate_and_print_interval = 10

        training = True
        train_epoch = max_epoch
        while training and epoch < train_epoch:  # loop over the dataset multiple times
        #for epoch in range(10):
            net.train()
            epoch += 1
            running_loss = 0.0
            correct = 0
            total = 0

            #cur_lr = get_adjusted_lr(epoch)
            #optimizer = optim.SGD(net.parameters(), cur_lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
            #print('Epoch {0}: {1:1.6f}'.format(epoch, cur_lr))

            print('Epoch: %d' % epoch)

            if (train_epoch > lr_total_reduce_times) and (epoch % (train_epoch // lr_total_reduce_times) == 0):
                lr /= 10
                optimizer = optim.SGD(net.parameters(), lr, momentum=0.9)
                print("Learning rate set to: {0:1.8f}".format(lr))

            mixup = False  # If use mixup or not

            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if gpu:
                    inputs, labels = inputs.to("cuda"), labels.to("cuda")
                else:
                    inputs, labels = inputs.to("cpu"), labels.to("cpu")

                # Mixup
                if mixup:
                    inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, 1.)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)
                if mixup:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # record the losses
                running_loss += loss.data.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                #num_loss += 1

                # print statistics
                if i % 100 == 0:  # print every 100 mini-batches
                    print(i, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                          % (running_loss / (i + 1), 100. * correct / total, correct, total))

                # print statistics
                #if i % 50 == 49:  # print every 200 mini-batches
                    #print('[%d, %4d] loss: %.3f' % (epoch, i + 1, running_loss / i))
            """
            print("Epoch {0:d}, Average loss:{1:.5f}".format(epoch, running_loss / num_loss))
            if (delta > 0):
                if ((abs(last_running_loss - running_loss)/num_loss < delta) or
                    (last_running_loss != 0) and (running_loss > last_running_loss)):
                    training = False
                    print("Stop trainning")
                    break;
                #print(abs(last_running_loss - running_loss))

            last_running_loss = running_loss
            running_loss = 0.0
            num_loss = 0
            """
            # print precision every 10 epoch

            if epoch % evaluate_and_print_interval == (evaluate_and_print_interval - 1):
                fitness_evaluate = eval_fitness(net, evaluateloader, 0, torch_batch_size, 0, gpu)
                fitness_test = eval_fitness(net, testloader, 0, torch_batch_size, 0, gpu)
                print('Epoch {3:d}: {0:3.3f}, {1:3.3f}, {2}'.format(fitness_evaluate, fitness_test, genome_id, epoch))
                ep = open("epoch.csv", "a")
                ep.write("{0}, {1:d}, {2:3.3f}, {3:3.3f}, {4:3.6f}\n".format(genome_id, epoch, fitness_evaluate, fitness_test, lr))
                ep.close()
            # reload run parameters

        print('Finished Training')

        # Write back parameters to NEAT
        net.write_back_parameters(genome)

        fitness_train = eval_fitness(net, trainloader, 0, torch_batch_size, 0, gpu)
        fitness_evaluate = eval_fitness(net, evaluateloader, 0, torch_batch_size, 0, gpu)
        fitness_test = eval_fitness(net, testloader, 0, torch_batch_size, 0, gpu)

        """

        print("Write parameters to genome.")
        net.write_back_parameters(genome)

        fitness_train = eval_fitness(net, trainloader, 0, torch_batch_size, 0, gpu)
        fitness_evaluate = eval_fitness(net, evaluateloader, 0, torch_batch_size, 0, gpu)
        fitness_test = eval_fitness(net, testloader, 0, torch_batch_size, 0, gpu)
        print('After write back: {0:3.3f}, {1:3.3f}, {2:3.3f}, {3}\n'.format(fitness_train, fitness_evaluate, fitness_test, genome_id))

        
        for module in net.children():
            for block in module:
                if isinstance(block, evaluate_torch.cnn_block):
                    #print(block.conv1.bias.data)
                    print(block.conv2.bias.data)
                elif isinstance(block, evaluate_torch.fc_block):
                    print(block.fc.bias.data)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        
        print("Create net from genome.")
        net2 = evaluate_torch.Net(config, genome, True)
       
        for module in net2.children():
            for block in module:
                if isinstance(block, evaluate_torch.cnn_block):
                    #print(block.conv1.bias.data)
                    print(block.conv2.bias.data)
                elif isinstance(block, evaluate_torch.fc_block):
                    print(block.fc.bias.data)
        
        if gpu:
            net2.cuda()

        fitness_train = eval_fitness(net2, trainloader, 0, torch_batch_size, 0, gpu)
        fitness_evaluate = eval_fitness(net2, evaluateloader, 0, torch_batch_size, 0, gpu)
        fitness_test = eval_fitness(net2, testloader, 0, torch_batch_size, 0, gpu)

        print('After reloaded: {0:3.3f}, {1:3.3f}, {2:3.3f}, {3}\n'.format(fitness_train, fitness_evaluate, fitness_test, genome_id))
        """

        #save the best net on test set
        if fitness_test > best_on_test_set:
            best_on_test_set = fitness_test
            torch.save(net, "best.pkl")

        best_every_generation.append((fitness_train, fitness_evaluate, fitness_test, genome_id, lr))

        genome.fitness = fitness_evaluate
        print('After: {0:3.3f}, {1:3.3f}, {2:3.3f}, {3}\n'.format(fitness_train, fitness_evaluate, fitness_test, genome_id))
        comp.write('{0:3.3f},{1:3.3f},{2:3.3f},{3},{4:3.6f}\n'.format(fitness_train, fitness_evaluate, fitness_test, genome_id, lr))
        comp.close
    if first_time:
        first_time = False

    best = 0.0
    best_id = 0
    for i in range(len(best_every_generation)):
        if best_every_generation[i][0] > best:
            best = best_every_generation[i][0]
            best_id = i

    res = open("result.csv", "a")
    res.write('{0:3.3f},{1:3.3f},{2:3.3f},{3},{4:3.6f}\n'.format(best_every_generation[best_id][0],
                                                                 best_every_generation[best_id][1],
                                                                 best_every_generation[best_id][2],
                                                                 best_every_generation[best_id][3],
                                                                 best_every_generation[best_id][4]))
    res.close()

    #del the net not in current population
    #net_id = set(net_dict.keys())
    #net_to_del = net_id - genomes_id_set
    #for genome_id in net_to_del:
    #    del net_dict[genome_id]

# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-mnist')

if torch.cuda.is_available():
    gpu = True
    print("Running on GPU!")
else:
    gpu = False
    print("Running on CPU!")

# reset result file
res = open("result.csv", "w")
info = open("info.txt", "w")
best = open("best.txt", "w")
res.close()
info.close()
best.close()
comp = open("comp.csv", "w")
comp.write("num,before,after_train,after_eva,after_test,id,lr,delta\n")
comp.close()
ep = open("epoch.csv", "w")
ep.write("id,epoch,eva,test,lr\n")
ep.close()

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))

# Run until a solution is found.
winner = p.run(eval_genomes)

# Run for up to 300 generations.
# pe = neat.ThreadedEvaluator(4, eval_genomes)
# winner = p.run(pe.evaluate)
# pe.stop()

# Display the winning genome.
#print('\nBest genome:\n{!s}'.format(winner))

"""
net = evaluate_torch.Net(config, winner)
if gpu:
    net.cuda()

final_train_epoch = 400

for epoch in range(final_train_epoch):

    # train the winner for some epoche
    lrfile = open("lr.txt", "r")
    tmp = lrfile.readline().rstrip('\n')
    lr = float(tmp)
    lrfile.close()

    criterion = nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
    optimizer = optim.SGD(net.parameters(), lr, momentum=0.9, weight_decay=5e-4)

    running_loss = 0.0
    correct = 0
    total = 0

    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        if gpu:
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
        else:
            inputs, labels = inputs.to("cpu"), labels.to("cpu")

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # record the losses
        running_loss += loss.data.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # print statistics
        if i % 50 == 49:  # print every 50 mini-batches
            print(i, len(trainloader), 'Epoch %d, Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (epoch, running_loss/(i+1), 100.*correct/total, correct, total))

print('Finished Final Training')

# save the model founded
torch.save(net, "model.pkl")
#net = torch.load("model.pkl")

evaluate_batch_size = 0
start = 0
fit = eval_fitness(net, testloader, evaluate_batch_size, torch_batch_size, start, gpu)

print("Final fitness: {0:3.3f}".format(fit))
"""
#TODO: wirte model to pytorch files

node_names = {# -28: '-28', -27: '-27', -26: '-26', -25: '-25', -24: '-24', -23: '-23', -22: '-22', -21: '-21',
              # -20: '-20', -19: '-19', -18: '-18', -17: '-17', -16: '-16', -15: '-15', -14: '-14', -13: '-13',
              # -12: '-12', -11: '-11', -10: '-10', -9: '-09', -8: '-08', -7: '-07', -6: '-06',
              -5: '-05', -4: '-04', -3: '-03', -2: '-02', -1: '-01', 0: '00', 1: '01', 2: '02',
              3: '03', 4: '04', 5: '05', 6: '06', 7: '07', 8: '08', 9: '09'}

# visualize.draw_net(config, winner, True, node_names=node_names)
