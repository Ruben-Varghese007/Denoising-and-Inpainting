import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the EnhancedCNN model
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 128x128x64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 128x128x128
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 128x128x256
        self.conv4 = nn.Conv2d(256, 3, kernel_size=3, padding=1)  # 128x128x3
        self.relu = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.batchnorm3 = nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))  # Retain size
        x = self.relu(self.batchnorm2(self.conv2(x)))  # Retain size
        x = self.relu(self.batchnorm3(self.conv3(x)))  # Retain size
        x = self.conv4(x)  # Output size matches input
        return x

# Function to resize the image
def resize_image(image, target_size=(128, 128)):
    return image.resize(target_size, Image.Resampling.LANCZOS)

# Function to run Deep Image Prior
def run_deep_image_prior(image_path):
    try:
        input_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    input_image = resize_image(input_image)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0).to(device)

    model = EnhancedCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)  # Smaller learning rate

    input_noise = torch.randn_like(input_image)

    patience = 100  # Increased patience
    best_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(2000):  # Increased the maximum number of epochs (2000)
        optimizer.zero_grad()
        output = model(input_noise)
        loss = nn.MSELoss()(output, input_image)
        loss.backward()
        optimizer.step()

        # Print loss every 100 epochs for monitoring
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/2000], Loss: {loss.item()}')

        if loss.item() < best_loss:
            best_loss = loss.item()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print("Early stopping...")
            break

    output_image = transforms.ToPILImage()(output.squeeze(0).cpu())

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Expected Output Image")
    plt.imshow(input_image.squeeze(0).permute(1, 2, 0).cpu())
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Current - Noisy Image")
    plt.imshow(output_image)
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = 'data/denoising/snail.jpg'  # Update this path to your image
    run_deep_image_prior(image_path)
