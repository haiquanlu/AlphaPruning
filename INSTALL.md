# Installation  
Step 1: Create a new conda environment:
```
conda create -n prune_llm python=3.9
conda activate prune_llm
```
Step 2: Install relevant packages
```
pip install torch==2.1.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.35.2 datasets==2.16.1 wandb sentencepiece
pip install accelerate==0.25.0
pip install weightwatcher

```