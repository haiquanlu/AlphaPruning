python main.py --model convnext_base \
    --data_path [PATH to ImageNet] \
    --resume [PATH to the pretrained weights] \
    --prune_metric [Prune metric] \
    --sparsity 0.8  \
    --save_dir [results save path] \
    --WW_metric alpha_mid \
    --epsilon 0.2