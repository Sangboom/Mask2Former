export CUDA_VISIBLE_DEVICES=0

# python train_net.py --num-gpus 8 \
#   --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_1x_ms.yaml --resume --eval-only

# python train_net.py --num-gpus 4 \
#   --config-file configs/cityscapes/instance-segmentation/maskformer2_R50_bs8_27k.yaml


# python train_net.py --num-gpus 4 \
#   --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml

python train_net.py --num-gpus 1 \
  --config-file configs/coco/instance-segmentation/dinov2/maskformer2_dinov2_base_adaptformer_bs16_50ep.yaml