# Q-ViT-DeiT
DeiT implementation for Q-ViT.

This code is built upon DeiT[https://github.com/facebookresearch/deit] and hustzxd's implementation[https://github.com/hustzxd/LSQuantization] for the LSQ paper.

### Environments

PyTorch 1.7.0+ and torchvision 0.8.1+ and pytorch-image-models 0.3.2
```shell
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

### How To Use
#### Train a float baseline using the script float_train.sh

For example:

```shell
python -m torch.distributed.launch \
--nproc_per_node=8 --use_env main.py \
--model deit_tiny_patch16_224_float \
--batch-size 256 \
--dist-eval \
--epochs 300 \
--output_dir path/to/float
```

#### Using unifrom_train.sh to train uniform-quantized qat model

For example:

```shell
wbits=4
abits=4
lr=5e-4
epochs=300
id=4bit_uniform

python -m torch.distributed.launch \
--nproc_per_node=8 --use_env main.py \
--model deit_tiny_patch16_224_mix \
--batch-size 64 \
--lr ${lr} \
--min-lr 0 \
--epochs ${epochs} \
--warmup-epochs 0 \
--wbits ${wbits} \
--abits ${abits} \
--dist-eval \
--output_dir path/to/uniform \
--finetune path/to/float
```

#### Using mixed_train.sh to train Q-ViT

For example:

```shell
wbits=5
abits=5
lr=2e-04
wd=0.05
epochs=300
lbd=1e-1
budget=21.455
id=4bit_mixed

python -m torch.distributed.launch \
--nproc_per_node=8 --use_env main.py \
--model deit_tiny_patch16_224_mix \
--batch-size 64 \
--lr ${lr} \
--min-lr 0 \
--epochs ${epochs} \
--weight-decay ${wd} \
--warmup-epochs 0 \
--wbits ${wbits} \
--abits ${abits} \
--bitops-scaler ${lbd} \
--budget ${budget} \
--stage-ratio 0.9 \
--dist-eval \
--mixpre \
--head-wise \
--output_dir path/to/mixed \
--finetune path/to/float
```
Here in Q-ViT, the arguments wbits and abits determines the initial bit-widths for weights and activations.
