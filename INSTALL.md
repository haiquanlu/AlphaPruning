# Installation  
Step 1: Create a new conda environment:
```
conda create -n prune_llm python=3.9
conda activate prune_llm
```
Step 2: Install relevant packages
```
pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.47.1 datasets==3.2.0 wandb sentencepiece
pip install accelerate
pip install weightwatcher

```