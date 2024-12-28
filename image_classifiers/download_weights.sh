mkdir -p model_weights/vit/
mkdir -p model_weights/convnext
mkdir -p model_weights/deit/

cd model_weights/vit
wget https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth
wget https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth

cd ../convnext/
wget https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth

cd ../deit 
wget https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth
wget https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth

cd ../..