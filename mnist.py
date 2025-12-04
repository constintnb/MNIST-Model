import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
from PIL import Image

# 1. 定义 CNN 模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2), #尺寸减半，变为14x14
                                          
                                         nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
                                         nn.ReLU(),
                                         nn.MaxPool2d(kernel_size=2) #尺寸再减半，变为7x7
                                         )
        
        self.flatten = nn.Flatten()

        self.fc_layers = nn.Sequential(nn.Linear(32*7*7, 128), #中间层
                                       nn.ReLU(),
                                       nn.Linear(128, 10)
                                       )

    def forward(self,x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


def train(dataloader, model, loss_fn, optimizer, device):

    model.train() # 训练模式
    
    running_loss = 0.0
    
    # 遍历数据
    for i, (imgs, targets) in enumerate(dataloader):
        # 搬运数据
        imgs, targets = imgs.to(device), targets.to(device)
        
        optimizer.zero_grad()         # 清零
        outputs = model(imgs)         # 预测
        loss = loss_fn(outputs, targets) # 算Loss
        loss.backward()               # 求梯度
        optimizer.step()              # 更新
        
        running_loss += loss.item()
        
        #打印日志 (每100个Batch打一次)
        if (i + 1) % 100 == 0:
            print(f"    [Batch {i+1}] Loss: {running_loss / 100:.4f}")
            running_loss = 0.0


def test(dataloader, model, loss_fn):

    model.eval() # 评估模式
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # 关闭梯度计算 (省内存，不反向传播)
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            # 预测
            outputs = model(imgs) # 输出是 logits (64, 10)
            
            # 计算 Loss (可选，用于观察)
            test_loss += loss_fn(outputs, targets).item()
            
            # 计算准确率
            pred = outputs.argmax(dim=1) # 沿着第 1 维度（即64），寻找最大值的索引
            correct += (pred == targets).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    
    print(f"测试结果: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct


if __name__ == "__main__":
    
    #2. 数据
    train_transform = transforms.Compose([
        # 数据增强
        transforms.RandomRotation(degrees=10),  # 随机旋转 ±10 度
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移 ±10%
        transforms.RandomCrop(28, padding=4),  # 随机裁剪，先填充 4 像素，再裁剪回 28x28

        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) 
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    #3. 损失函数和优化器

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    #4. 训练、测试

    epochs = 5

    best_ac = 0
    
    print(f"开始训练，使用设备: {device}")
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # 调用训练函数 
        train(train_loader, model, loss_fn, optimizer, device)
        
        # 调用测试函数 
        ac=test(test_loader, model, loss_fn)

        # 保存准确率最高的模型
        if ac>best_ac:
            best_ac=ac
            torch.save(model.state_dict(), 'best_mnist_cnn.pth')
            print(f"保存当前最佳模型，准确率: {best_ac*100:.2f}%")

        
    print("所有任务完成！")
    
