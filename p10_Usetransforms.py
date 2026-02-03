from csv import writer
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter


img = Image.open("dataset/train/ants_image/0013035.jpg")

writer = SummaryWriter("logs")

# 将PIL图像转换为Tensor图像
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("tensor", img_tensor, 0)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("norm", img_norm, 1)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_totensor(img_resize)
writer.add_image("resize", img_resize, 2)
print(img_resize)

# # # Compose - resize - 2
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("resize_2", img_resize_2, 3)

# RandomCrop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)



writer.close()