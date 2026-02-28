from torch.utils.tensorboard.writer import SummaryWriter
import torchvision

from torch.utils.data import DataLoader

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 测试数据集中第一张图片及其标签
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloder")
step = 0

for epoch in range(2):
    for data in test_loader:
        imgs, targets = data
        writer.add_images(f"epoch:{epoch}_shuffle", imgs, step)
        step = step + 1

writer.close()