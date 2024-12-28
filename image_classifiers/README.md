# Pruning Image Classifiers
Here we provide the code for pruning ConvNeXt and ViT. 

## Environment
We additionally install `timm` for loading pretrained image classifiers.
```
pip install timm==0.4.12
```

## Download Weights
Run the script [download_weights.sh](download_weights.sh) to download pretrained weights for ConvNeXt-B, DeiT-B, DeiT-s, ViT-B and ViT-L.

## Usage
Here is the command for pruning ConvNeXt/ViT/DeiT models:
```
python main.py --model convnext_base \
    --data_path [PATH to ImageNet] \
    --resume [PATH to the pretrained weights] \
    --prune_metric [Prune metric] \
    --sparsity 0.8  \
    --save_dir [results save path] \
    --WW_metric alpha_mid \
    --epsilon 0.2
```

where:
- `--model`: network architecture, choices [`convnext_base`, `deit_small_patch16_224`, `deit_base_patch16_224`, `vit_large_patch16_224`, `vit_base_patch16_224`].
- `--resume`: model path to downloaded pretrained weights.
- `--prune_metric`: [`magnitude`, `wanda`, `magnitude_ww`, `wanda_ww`].
- `--WW_metric`: the HT-SR metric type to use.
- `--epsilon`: a hyperparamter to adjust the layerwise sparsity.