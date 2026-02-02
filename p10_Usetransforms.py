from csv import writer
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


img_path = Image.open("dataset/train/ants_image/0013035.jpg")

writer = SummaryWriter("logs")

# 将PIL图像转换为Tensor图像
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img_path)
writer.add_image("tensor", img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("norm", img_norm)

writer.close()