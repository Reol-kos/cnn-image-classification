import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from enhanced_model import EnhancedCIFAR10Model  # 如果模型定义在单独文件中，记得导入

# 加载测试集
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EnhancedCIFAR10Model()
model.load_state_dict(torch.load('./data/enhanced_model.pth', map_location=device))
model.to(device)
model.eval()

# 测试模型性能
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the enhanced model on the CIFAR-10 test images: {100 * correct / total:.2f}%')
