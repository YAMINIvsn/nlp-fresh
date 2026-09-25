import torch
import transformers
import sklearn
import shap

print("All libraries loaded successfully")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")