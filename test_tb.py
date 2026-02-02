from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")

image_path = "dataset/train/bees_image/21399619_3e61e5bb6f.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

writer.add_image("train", img_array, 1, dataformats="HWC")

for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

# writer.add_scalar()

writer.close()