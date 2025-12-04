import torch
from torchvision import transforms
from PIL import Image
from torch import nn
import torchvision.transforms.functional as TF

from mnist import CNN  # 引用模型

# 预测函数
def predict_digit(image_path, model_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载模型结构与权重
    model = CNN().to(device)
    
    # 加载训练好的权重文件 (.pth)
    # map_location 保证即使你在 GPU 训练，在只有 CPU 的电脑上也能跑
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 评估模式
    model.eval()

    # 图片预处理 (必须和训练时保持一致) 
    # 1. 变成灰度图 (L)
    # 2. 缩放到 28x28
    # 3. 转 Tensor 并 归一化
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1), # 强制转单通道
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 读取图片并处理
    try:
        img = Image.open(image_path).convert('RGB')
        img = TF.invert(img)  # 反色处理，黑底白字转白底黑字
        
        # 预处理
        img_tensor = transform(img)
        
        # 增加 Batch 维度: (1, 28, 28) -> (1, 1, 28, 28)
        # 因为模型只吃 Batch
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # 预测 
        with torch.no_grad(): # 关闭梯度计算
            output = model(img_tensor)
            
            # 获取概率最大的索引
            pred_idx = output.argmax(dim=1).item()
            
            # (可选) 获取置信度
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence = probs[0][pred_idx].item()

        print(f"---------------------------")
        print(f"图片: {image_path}")
        print(f"预测结果: 【 {pred_idx} 】")
        print(f"置信度: {confidence:.2%}")
        print(f"---------------------------")

    except Exception as e:
        print(f"错误: 无法读取或处理图片. {e}")


if __name__ == '__main__':
    
    img_path = 'num1.png' 
    model_file = 'best_mnist_cnn.pth'
    
    # 只有当图片文件存在时才运行
    import os
    if os.path.exists(img_path) and os.path.exists(model_file):
        predict_digit(img_path, model_file)
    else:
        print("图片不存在")