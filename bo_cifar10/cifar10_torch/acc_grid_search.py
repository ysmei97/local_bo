import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np
from scipy.stats import norm
from sklearn.metrics.pairwise import rbf_kernel

from models import *
from utils import progress_bar

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_ids = [0]

dimension = 2
iteration = 20
layer_sample = 6
width_sample = 40
# model_sample = 21
# batch_size_sample = 5
train_epoch_offset = 100
x_star_num = 3
T = 12
epsilon = 0.1
neighbor_threshold = 0.8
alpha = 10
length_scale = 0.1


def model_train(xx):
    """Tunable hyperparameter (within xx) for searching local optima
        Global Arguments:
            model: network type
            batch_size: batch size
            learning_rate: learning rate
        Local Arguments (ResNeXt):
            layer: network size
            cardinality: number of group convolution
            width: width of the bottleneck
    """
    model = 'ResNeXt'
    layer = int(xx[0])
    batch_size = 128
    cardinality = 4
    width = int(xx[1])

    # best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    '''
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    '''

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=int(batch_size), shuffle=True, num_workers=2)

    '''
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2)
    '''

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    '''
    model_name = ['VGG11', 'VGG13', 'VGG16', 'VGG19', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
                  'GoogLeNet', 'DenseNet121', 'DenseNet169', 'DenseNet201', 'MobileNet', 'MobileNetV2', 'DPN26',
                  'DPN92', 'ResNeXt29_2x64d','ResNeXt29_4x64d','ResNeXt29_8x64d', 'ResNeXt29_32x4d']
    model_type = [VGG('VGG11'), VGG('VGG13'), VGG('VGG16'), VGG('VGG19'), ResNet18(), ResNet34(), ResNet50(),
                  ResNet101(), ResNet152(), GoogLeNet(), DenseNet121(), DenseNet169(), DenseNet201(), MobileNet(),
                  MobileNetV2(), DPN26(), DPN92(), ResNeXt29_2x64d(), ResNeXt29_4x64d(), ResNeXt29_8x64d(), 
                  ResNeXt29_32x4d()]
    '''
    model_size = ['29', '38', '50', '68', '86', '101']
    num_blocks = [[3, 3, 3], [4, 4, 4], [3, 4, 6, 3], [3, 4, 12, 3], [3, 4, 18, 3], [3, 4, 23, 3]]
    print('==> Using network: %s, ' % model, 'layer: %s, ' % model_size[layer], 'cardinality: %d, ' % cardinality,
          'width: %d' % width)
    # net = model_type[model]
    net = ResNeXt_(num_blocks[layer], cardinality, width)
    # net = VGG(model_size[layer])
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        # best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    train_accuracy_epoch = []
    for epoch in range(start_epoch, start_epoch + train_epoch_offset):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                         (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        train_accuracy_epoch.append(100. * correct / total)
    return max(train_accuracy_epoch)


'''
x_star = []
for _ in range(x_star_num):
    x_star.append(np.array([int(np.random.uniform(0, 4)), int(np.random.uniform(5, 11))]))
'''
# x_star = [np.array([0, 4]), np.array([1, 34]), np.array([3, 64])]
# print('Initial inputs: ', x_star)
x1_sample = np.linspace(0, 5, layer_sample)
x2_sample = np.linspace(5, 44, width_sample)
x_sample = np.meshgrid(x1_sample, x2_sample)

for i in range(width_sample):
    for j in range(layer_sample):
        xx = np.array([x_sample[0][i][j], x_sample[1][i][j]])
        train_accuracy = model_train(xx)
        with open('search_cifar', 'a') as output:
            print('\nInputs:', xx, 'value:', train_accuracy, '\n', file=output)
            output.close()
        sheet.cell(row=i+1, column=j+1).value = train_accuracy

