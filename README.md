# Denoising and Inpainting
### An Implementation of Deep Image Prior for Denoising and Inpainting.

->**Deep Image Prior**
- Deep Image Prior Repo : [ https://github.com/DmitryUlyanov/deep-image-prior.git ]

-> **Features**
- **Image Denoising**: Removes noise from corrupted images using DIP.
- **Image Inpainting**: Fills missing parts of an image using learned priors.
- **GPU Acceleration**: Supports CUDA and cuDNN for efficient computation.
- **Web Interface**: Uses **Streamlit** for an interactive user interface.

## Instructions

### Clone the Repository (Terminal)
- Navigate to your project directory
  ```sh
  cd path_to_your_project
  ```
- Clone the **IoT_Object_Detection_YOLOv5** repository
  ```sh
  git clone https://github.com/Ruben-Varghese007/Denoising-and-Inpainting.git
  ```
  
### To run on Command Prompt
- Open cmd

### Navigate to File Location
- Allows for change of the current working directory
```sh
cd
```
- For example - if the file is present in F Drive
```sh
F:
```
- Navigate to the deep image prior directory - F:\deep-image-prior
```sh
cd deep-image-prior
```

### Activate Virtual Environment
```sh
conda activate deep-image-prior
```

### Run Code
```sh
python filename.py
```
-> Example - python run_dip_denoise.py OR python run_dip_inpaint.py

### To Deploy Denoising on Streamlit
```sh
streamlit run deploy.py
```

## Requirements

### Install required libraries:

**PyTorch**:

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117  # Ensure correct CUDA version
```

**Streamlit**:

```sh
pip install streamlit
```

**Others**:
```sh
pip install numpy matplotlib tqdm opencv-python
```

Ensure you have **CUDA** and **cuDNN** installed and properly configured for GPU acceleration.

### **Install Dependencies**

#### **1. Setup CUDA & cuDNN (For GPU Support)**
- Install **NVIDIA CUDA Toolkit** (Recommended: CUDA 11.7+)
- Install **cuDNN** and configure environment variables
- Verify installation using:

  ```sh
  nvcc --version  # Check CUDA version
  ```

### Recommendation

- It is highly recommended to run the programs on a GPU for better performance and efficiency.

## Acknowledgments
This project is based on **[Deep Image Prior](https://github.com/DmitryUlyanov/deep-image-prior)** by **Dmitry Ulyanov**, licensed under **Apache 2.0**. Some resources from the original repository have been used in this project.

## License
This project is licensed under the **Apache License 2.0**. Please refer to the [LICENSE](LICENSE) file for details.

**Note:** The original author requests to be contacted for commercial use of this software.

