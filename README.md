# Denoising and Inpainting
### An Implementation of Deep Image Prior for Denoising and Inpainting.

**Deep Image Prior**
- An Implementation of Deep Image Prior for Denoising and Inpainting
- Deep Image Prior Repo : [ https://github.com/DmitryUlyanov/deep-image-prior.git ]

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
> Open cmd

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

**PyTorch**:

```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117  # Ensure correct CUDA version
```

**Streamlit**:

```sh
pip install streamlit
```

- Recommend to Run the programs on GPU
