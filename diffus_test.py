import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import torch.multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class ConditionalUNet(nn.Module):
    def __init__(self, num_classes, in_channels=1, out_channels=1):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = 10
        self.class_embedding = nn.Embedding(self.num_classes, self.embedding_dim)
        
         # Нисходящий путь
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
             nn.ReLU(),
            nn.MaxPool2d(2)  # 28x28 -> 14x14
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 14x14 -> 7x7
        )
        
        # Средний слой
        self.mid = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU()
        )
        
        # Восходящий путь
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 7x7 -> 14x14
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 14x14 -> 28x28
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU()
        )
        
        self.final = nn.Conv2d(32, out_channels, 3, padding=1)
        
        # Линейный слой для встраивания классов
        self.class_proj = nn.Linear(self.embedding_dim, 128)
        
    def forward(self, x, class_labels):
        # Встраивание класса
        emb = self.class_embedding(class_labels)
        emb = self.class_proj(emb).view(emb.size(0), 128, 1, 1)
        
        # Нисходящий путь
        x = self.down1(x)  # 32x14x14
        x = self.down2(x)  # 64x7x7
        
        # Средний слой с добавлением класса
        x = self.mid(x)    # 128x7x7
        x = x + emb  # Добавляем информацию о классе
        
        # Восходящий путь
        x = self.up1(x)    # 64x14x14
        x = self.up2(x)    # 32x28x28
        
        return self.final(x)  # 1x28x28

# Создайте экземпляр модели
num_classes = 4  # Должно совпадать с обучением
model = ConditionalUNet(num_classes).to(device)

# Укажите путь к файлу
checkpoint_path = "diffusion_model_epoch_500.pth"

# Загрузка чекпоинта
checkpoint = torch.load(checkpoint_path, map_location=device)

# Восстановление модели
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Переключение в режим инференса

def corrupt(x, amount):
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)
    return x * (1 - amount) + noise * amount

def generate_image(model, class_idx, steps=40):
    model.eval()
    with torch.no_grad():
        # Создаем шум
        x = torch.randn(1, 1, 28, 28).to(device)
        labels = torch.tensor([class_idx], dtype=torch.long).to(device)
        
        # Постепенная дениризация
        for i in range(steps):
            noise_amount = torch.tensor([1 - i/steps]).to(device)
            noisy_x = corrupt(x, noise_amount)
            pred = model(noisy_x, labels)
            x = 0.7 * x + 0.3 * pred
        
        # Преобразуем в изображение
        img = x.cpu().squeeze().numpy()  # Инверсия цветов
        return img

# Генерация изображения
class_idx = int(input())  # Индекс класса (0-3)
image = generate_image(model, class_idx, steps=15)

# Отображение
plt.imshow(image, cmap="gray")
#plt.title(f"Сгенерированное изображение класса: {checkpoint['classes'][class_idx]}")
plt.axis('off')
plt.show()