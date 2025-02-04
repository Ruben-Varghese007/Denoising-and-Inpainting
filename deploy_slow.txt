# Slower Process of Execution

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

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
def run_deep_image_prior(input_image, noisy_image):
    input_image = input_image.to(device)
    noisy_image = noisy_image.to(device)

    model = EnhancedCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

    patience = 100
    best_loss = float('inf')
    early_stopping_counter = 0
    loss_values = []

    for epoch in range(2000):
        optimizer.zero_grad()
        output = model(noisy_image)

        # Ensure the output is resized to match input image
        if output.size() != input_image.size():
            output = torch.nn.functional.interpolate(output, size=input_image.shape[2:])

        loss = nn.MSELoss()(output, input_image)
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())

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

    return output, loss_values

# Streamlit app to interact with the model
def main():
    st.title("Deep Image Prior - Image Denoising")

    # Upload image
    image_file = st.file_uploader("Choose an Image...", type=["jpg", "png", "jpeg"])

    if image_file is not None:
        input_image = Image.open(image_file).convert('RGB')
        input_image_resized = resize_image(input_image)

        # Upload noise image (optional)
        noise_file = st.file_uploader("Choose a Noise Image (Optional)...", type=["jpg", "png", "jpeg"])

        if noise_file is not None:
            noise_image = Image.open(noise_file).convert('RGB')
            noise_image = resize_image(noise_image)
            noise_tensor = transforms.ToTensor()(noise_image).unsqueeze(0)
        else:
            # Generate random noise
            noise_tensor = torch.randn(1, 3, 128, 128).to(device)

        # Combine input image and noise (noisy image)
        input_image_resized = transforms.ToTensor()(input_image_resized).unsqueeze(0)
        noisy_image = input_image_resized + noise_tensor

        if st.button("Run Deep Image Prior"):
            # Run the deep image prior model
            output_image, loss_values = run_deep_image_prior(input_image_resized, noisy_image)

            # Clamp the output to ensure values are between 0 and 1
            output_image = torch.clamp(output_image, 0, 1)

            # Convert the output tensor to an image
            output_image = transforms.ToPILImage()(output_image.squeeze(0).cpu())
            noisy_image_display = transforms.ToPILImage()(noisy_image.squeeze(0).cpu())

            # Display input, noise, noisy, and output images
            st.write("### Input Image")
            st.image(input_image, caption='Input Image', use_column_width=True)

            st.write("### Noise Added")
            st.image(noise_image, caption='Noise Image', use_column_width=True)

            st.write("### Noisy Image (Input + Noise)")
            st.image(noisy_image_display, caption='Noisy Image', use_column_width=True)

            st.write("### Restored Image")
            st.image(output_image, caption='Restored Image', use_column_width=True)

            # Display the loss graph
            st.write("### Model Training Loss")
            st.line_chart(loss_values)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
