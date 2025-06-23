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
data_dir = "C:/Users/user/Desktop/tren"
image_size = 28
n_epochs = 500

# Трансформации
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
])
# Создание датасета
full_dataset = datasets.ImageFolder(
    root=data_dir,
    transform=transform
)
# DataLoader
train_dataloader = DataLoader(
    full_dataset,
    batch_size=8,  # Уменьшим размер батча
    shuffle=True,
    num_workers=0,
    pin_memory=True if device.type == "cuda" else False
)
print(f"Размер датасета: {len(full_dataset)} изображений")
print(f"Классы: {full_dataset.classes}")
num_classes = len(full_dataset.classes)

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

# Инициализация модели
net = ConditionalUNet(num_classes=num_classes)
net.to(device)
print("Количество параметров:", sum(p.numel() for p in net.parameters()))

def corrupt(x, amount):
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)
    return x * (1 - amount) + noise * amount

# Обучение модели
def tren():
    # Функции потерь и оптимизатор
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = []
    
    for epoch in range(n_epochs):
        net.train()
        epoch_losses = []
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for x, y in progress_bar:
            x = x.to(device)
            y = y.to(device)
            
            # Добавление шума
            noise_amount = torch.rand(x.shape[0]).to(device)
            noisy_x = corrupt(x, noise_amount)
            
            # Предсказание
            pred = net(noisy_x, y)
            
            # Вычисление потерь
            loss = loss_fn(pred, x)
            
            # Оптимизация
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Сохранение лоссов
            losses.append(loss.item())
            epoch_losses.append(loss.item())
            
            # Обновление прогресс-бара
            progress_bar.set_postfix({"Loss": loss.item()})
        
        # Вычисление среднего лосса за эпоху
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1}/{n_epochs} | Avg Loss: {avg_epoch_loss:.5f}")
        
        # Сохранение модели
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
            'classes': full_dataset.classes,
        }, f"diffusion_model_epoch_{epoch+1}.pth")
        print(f"Model saved at epoch {epoch+1}")

    # График потерь
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("training_loss.png")
    plt.show()
tren()

# Функция генерации образцов
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

# Генерация примеров
def test_image(model):
    fig, axes = plt.subplots(1, num_classes, figsize=(15, 3))
    for class_idx in range(num_classes):
        samples = generate_image(model, class_idx, 15)
        ax = axes[class_idx] if num_classes > 1 else axes
        ax.imshow(samples, cmap="gray")
        ax.set_title(f"Class: {full_dataset.classes[class_idx]}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("generated_samples.png")
    plt.show()
test_image(net)