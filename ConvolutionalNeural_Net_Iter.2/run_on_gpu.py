"""
Runs a PyTorch script on the GPU without changing the script itself.
It makes the GPU PyTorch's default device, so every model and tensor the script creates lands on the GPU.

Usage:  python run_on_gpu.py CNN_iteration_2_Through_Focus.py

Meant for training. Evaluation code that calls .numpy() on model outputs needs CPU tensors,
so run those parts with plain `python <script>.py` instead.
"""
import os
import runpy
import sys

import torch

if len(sys.argv) < 2:
    sys.exit('Usage: python run_on_gpu.py <script.py> [args...]')
if not torch.cuda.is_available():
    sys.exit('PyTorch cannot see a CUDA GPU - is the CUDA build of torch installed? (torch.__version__ = ' + torch.__version__ + ')')

torch.set_default_device('cuda')
print(f'run_on_gpu: using {torch.cuda.get_device_name(0)}')

script = os.path.abspath(sys.argv[1])
sys.argv = [script] + sys.argv[2:]                  # the script sees its own name + args, as if run directly
sys.path.insert(0, os.path.dirname(script))         # so it can import files next to it
runpy.run_path(script, run_name='__main__')

# Run the following in terminal with: python run_on_gpu.py CNN_iteration_2_Through_Focus.py
# c:/Users/CJ/Documents/GitHub/SEM_2026-2027/ConvolutionalNeural_Net_Iter.2/run_on_gpu.py 
# c:/Users/CJ/Documents/GitHub/SEM_2026-2027/ConvolutionalNeural_Net_Iter.2/CNN_iteration_2_Through_Focus.py