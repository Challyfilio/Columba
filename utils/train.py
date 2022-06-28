import torch
import numpy as np


def rightness(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()  # 将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return rights, len(labels)  # 返回正确的数量和这一次一共比较了多少元素


def train_net(args, net, train_loader, val_loader, criterion, optimizer, device):
    record = []  # 记录准确率等数值的容器
    net.train(True)
    best_r = 0.0
    for epoch in range(args.epochs):
        train_rights = []  # 记录训练数据集准确率的容器
        train_losses = []
        for batch_idx, (data, target) in enumerate(train_loader):  # 针对容器中的每一个批进行循环
            data, target = data.clone().detach().requires_grad_(False), target.clone().detach()
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss = criterion(output, target)  # 计算误差
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 一步随机梯度下降
            right = rightness(output, target)  # 计算准确率所需数值，返回正确的数值为（正确样例数，总样本数）
            train_rights.append(right)  # 将计算结果装到列表容器中
            loss = loss.cpu()
            train_losses.append(loss.data.numpy())

        # 分别记录训练集中分类正确的数量和该集合中总的样本数
        train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))

        # 在测试集上分批运行，并计算总的正确率
        net.eval()
        vals = []
        # 对测试数据集进行循环
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            data, target = data.clone().detach().requires_grad_(True), target.clone().detach()
            output = net(data)
            val = rightness(output, target)  # 获得正确样本数以及总样本数
            vals.append(val)

        # 计算准确率
        val_r = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
        val_ratio = 1.0 * val_r[0].cpu().numpy() / val_r[1]

        if val_ratio > best_r:
            best_r = val_ratio
            torch.save(net.state_dict(), args.output_dir + '/best_model.pth')
        # 打印准确率等数值，其中正确率为本训练周期Epoch开始后到目前撮的正确率的平均值
        print('epoch: {}\t'
              'loss: {:.6f}\t\t'
              'train_acc: {:.2f}%\t'
              'val_acc: {:.2f}%'.format(
            epoch,
            np.mean(train_losses),
            100. * train_r[0].cpu().numpy() / train_r[1],
            100. * val_r[0].cpu().numpy() / val_r[1]
        )
        )
        record.append(
            [epoch, np.mean(train_losses), train_r[0].cpu().numpy() / train_r[1], val_r[0].cpu().numpy() / val_r[1]])
    return record
