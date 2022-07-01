import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import accuracy_score, classification_report


def image_show(args, data_loader, class_name, count=1):
    images, labels = next(iter(data_loader))
    plt.clf()
    plt.figure(figsize=(12, 8))
    for i in images:
        plt.subplot(4, 8, count)
        picture_show = np.transpose(i.numpy(), (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        picture_show = std * picture_show + mean
        picture_show = picture_show / np.amax(picture_show)
        picture_show = np.clip(picture_show, 0, 1)
        plt.imshow(picture_show)
        plt.title(class_name[int(labels[count - 1].numpy())])
        plt.xticks([])
        count += 1
        plt.axis("off")
    plt.show()
    plt.savefig(args.output_dir + '/imshow.png')


def curve_draw(args, record):
    sns.set_theme(style="whitegrid")
    epoch = [record[0] for record in record]
    loss = [record[1] for record in record]
    acc = [record[2] for record in record]
    plt.clf()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(epoch, acc, label='acc')
    plt.plot(epoch, loss, label='loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    plt.savefig(args.output_dir + '/acc_loss.png')


def val_and_visualize(args, net, val_loader, class_name, device):
    images, labels = next(iter(val_loader))
    net.load_state_dict(torch.load(args.output_dir + '/best_model.pth'))
    net.to(device)
    net.eval()

    with torch.no_grad():
        images = images.to(device)
        outputs = net(images)
        _, preds = torch.max(outputs, 1)

    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    images = images.cpu().numpy()

    print('\nacc is: {:.2f}%\n'.format(100. * accuracy_score(preds, labels)))
    print(classification_report(preds, labels))

    # 可视化模型预测
    plt.clf()
    plt.figure(figsize=(12, 8))
    for i in range(0, 32):
        plt.subplot(5, 8, i + 1)
        color = 'blue' if preds[i] == labels[i] else 'red'
        plt.title(class_name[preds[i]], color=color)
        picture_show = np.transpose(images[i], (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        picture_show = std * picture_show + mean

        picture_show = picture_show / np.amax(picture_show)
        picture_show = np.clip(picture_show, 0, 1)
        plt.imshow(picture_show)
        plt.axis('off')
    plt.show()
    plt.savefig(args.output_dir + '/visualize.png')
