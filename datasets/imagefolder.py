import torch.utils.data as tud
from torchvision.datasets import ImageFolder
from torchvision import transforms


root = "./data"

transform = transforms.Compose([
    transforms.ToTensor()
])

datasets = ImageFolder(root, transform=transform)
dataloader = tud.DataLoader(datasets, batch_size=2, shuffle=True)

for i, (img, target) in enumerate(dataloader):
    print(img.shape)
    print(target)