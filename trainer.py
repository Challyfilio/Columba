import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms

from utils.train import train_net
from utils.tools import image_show, curve_draw, val_and_visualize


def create_dataset(data_dir, training=True):
    image_size = 224
    if training:
        trans = [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]
    else:
        trans = [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]
    dataset = datasets.ImageFolder(os.path.join(data_dir), transforms.Compose(trans))
    return dataset


def create_dataloader(args):
    if args.dataset == 'aircraft':
        train_data_dir = 'dataset/aircraft/train'
        val_data_dir = 'dataset/aircraft/test'
        class_name = {0: "747-400", 1: "A380", 2: "Challenger 600", 3: "Eurofighter Typhoon", 4: "F_A-18",
                      5: "Hawk T1", 6: "Metroliner", 7: "SR-20", 8: "Tornado", 9: "Tu-154"}
    elif args.dataset == 'aircraft_similar':
        train_data_dir = 'dataset/aircraft_similar/train'
        val_data_dir = 'dataset/aircraft_similar/test'
        class_name = {0: "737-200", 1: "737-300", 2: "737-400", 3: "737-500", 4: "737-600",
                      5: "737-700", 6: "737-800", 7: "737-900", 8: "777-200", 9: "777-300"}
    else:
        print('dataset error')
        exit()

    train_dataset = create_dataset(train_data_dir, training=True)
    val_dataset = create_dataset(val_data_dir, training=False)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=40, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=334, shuffle=True, num_workers=8)

    return train_loader, val_loader, class_name


def load_net(args):
    # 检测是否有GPU
    use_cuda = torch.cuda.is_available()

    # 当GPU可用时
    device = 'cuda:2' if use_cuda else 'cpu'

    if args.net == 'resnet18':
        # 加载resnet 加载预训练权重
        net = models.resnet18(pretrained=False)
        net.load_state_dict(torch.load('pretrain_model/resnet18-5c106cde.pth'))
    elif args.net == 'resnet50':
        net = models.resnet50(pretrained=False)
        net.load_state_dict(torch.load('pretrain_model/resnet50-19c8e357.pth'))
    else:
        print('net error')
        exit()

    net = net.to(device)

    # 冻结除最后一层外的参数
    if args.only_train_linear:
        for param in net.parameters():
            param.requires_grad = False

    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 10)  # 修改线形层输出
    net.fc = net.fc.to(device)
    return net, device


def train(args):
    train_loader, val_loader, class_name = create_dataloader(args)
    image_show(args, train_loader, class_name)
    net, device = load_net(args)

    criterion = nn.CrossEntropyLoss()  # Loss
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)  # 优化器
    record = train_net(args, net, train_loader, val_loader, criterion, optimizer, device)

    curve_draw(args, record)

    val_and_visualize(args, net, val_loader, class_name, device)


def test(args):
    train_loader, val_loader, class_name = create_dataloader(args)
    net, device = load_net(args)
    val_and_visualize(args, net, val_loader, class_name, device)
