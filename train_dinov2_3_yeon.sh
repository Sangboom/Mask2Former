export CUDA_VISIBLE_DEVICES=2

# python train_net.py --num-gpus 8 \
#   --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_1x_ms.yaml --resume --eval-only

# python train_net.py --num-gpus 4 \
#   --config-file configs/cityscapes/instance-segmentation/maskformer2_R50_bs8_27k.yaml


# python train_net.py --num-gpus 4 \
#   --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml

# python train_net.py --num-gpus 4 \
#   --config-file configs/coco/instance-segmentation/dinov2/maskformer2_dinov2_base_bs16_50ep.yaml --dist-url='tcp://127.0.0.1:8475'


python train_net.py --num-gpus 1 \
  --config-file configs/coco/instance-segmentation/dinov2/maskformer2_dinov2_base_freeze_bs16_50ep.yaml --dist-url='tcp://127.0.0.1:8477'