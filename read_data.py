
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np

writer = SummaryWriter("logs")

class MyDataset(Dataset):

    def __init__(self, root_dir, img_dir, label_dir, transform):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_path = os.path.join(self.root_dir, self.img_dir)
        self.label_path = os.path.join(self.root_dir, self.label_dir)
        self.img_list = os.listdir(self.img_path)
        self.label_list = os.listdir(self.label_path)
        self.img_list.sort()
        self.label_list.sort()

        self.transform = transform

    def __getitem__(self, index):

        image_name = self.img_list[index]
        img_item_path = os.path.join(self.img_path, image_name)
        img = Image.open(img_item_path) 
        label_name = self.label_list[index]
        label_item_path = os.path.join(self.label_path, label_name)
        
        with open(label_item_path, 'r') as f:
            label = f.readline()

        img = self.transform(img)

        sample = {
            'image': img,
            'label': label
        }

        return sample

    def __len__(self):
        
        assert len(self.img_list) == len(self.label_list)
        return len(self.img_list)

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

root_dir = "dataset/train"
ants_image_dir = "ants_image"
bees_image_dir = "bees_image"
ants_label_dir = "ants_label"
bees_label_dir = "bees_label"

ants_dataset = MyDataset(root_dir, ants_image_dir, ants_label_dir, transform)
bees_dataset = MyDataset(root_dir, bees_image_dir, bees_label_dir, transform)

train_dataset = ants_dataset + bees_dataset

dataloader = DataLoader(train_dataset, batch_size=1, num_workers=2)

    # 取出第一张图像及其标签
first_sample = train_dataset[0]
first_img = first_sample['image']      # shape: [C, H, W]
first_label = first_sample['label']

# 用 matplotlib 显示第一张图片
plt.figure()
plt.imshow(np.transpose(first_img.numpy(), (1, 2, 0)))  # 从 CHW -> HWC
plt.axis("off")
plt.title(first_label)
plt.show()

# 同时把第一张图片写入 TensorBoard，方便在浏览器中查看
writer.add_image('first_image', first_img)
writer.close()



