# PyTorch on Apple Silicon
How to install PyTorch on Apple M chips to use Apple Metal (GPU)

This guide is a personal note that I get out of the outstanding [Daniel Bourke's tutorial](https://www.youtube.com/watch?v=Zx2MHdRgAIc) 

## Requirements
- Apple Silicon Mac (M1 or M2, at the time of writing)
- MacOS 12.3+ (PyTorch will work on previous versions, but the GPU on your Mac won't get used)

## Steps
1. Download [Miniforge3](https://github.com/conda-forge/miniforge#miniforge-pypy3) for macOS arm64 chips. You can also choose your preferred package management 
to install Miniforge3 into home directory:

```
chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
source ~/miniforge3/bin/activate
```

3. Restart terminal

4. Create a directory test PyTorch

5. Create and activate Conda test env env
   
```conda create --name YOUR_ENV_NAME python=3.11.3```

7. Install PyTorch
   
```pip3 install torch torchvision torchaudio```

9. Install Jupyter (optional)
```pip install jupyter```

10. Other data science packages:
```pip install jupyter pandas numpy matplotlib scikit-learn tqdm```

11. Run the following snippet (via Jupyter Notebook, if you installed it) to verify that PyTorch is running on Apple Metal (GPU) 
```python
import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}")

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device      
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Create data and send it to the device
x = torch.rand(size=(3, 4)).to(device)
print(x)
```

If you see an output similar to the one below, you are good to go
```
PyTorch version: 2.0.1
Is MPS (Metal Performance Shader) built? True
Is MPS available? True
Using device: mps
tensor([[0.5628, 0.8255, 0.0803, 0.1202],
        [0.5967, 0.3287, 0.2962, 0.6132],
        [0.7556, 0.2290, 0.9701, 0.8275]], device='mps:0')
```



## Using in your code
To run data/models on an Apple Silicon (GPU), use the PyTorch device name "mps" with .to("mps"). MPS stands for Metal Performance Shaders, Metal is Apple's GPU framework.

```python
import torch

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Create data and send it to the device
x = torch.rand(size=(3, 4)).to(device)
x
```

If you see an output similar to the one below, you're good to go.


```
tensor([[2.6020e-01, 9.6467e-01, 7.5282e-01, 1.8063e-01],
        [7.0760e-02, 9.8610e-01, 6.5195e-01, 7.5700e-01],
        [3.4065e-01, 1.8971e-01, 6.0876e-01, 9.3907e-01]], device='mps:0')v
```
 
