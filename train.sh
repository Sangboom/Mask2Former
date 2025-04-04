export CUDA_VISIBLE_DEVICES=0,1,2,3

# python train_net.py --num-gpus 8 \
#   --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_1x_ms.yaml --resume --eval-only

# python train_net.py --num-gpus 4 \
#   --config-file configs/cityscapes/instance-segmentation/maskformer2_R50_bs8_27k.yaml


# python train_net.py --num-gpus 4 \
#   --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml

python train_net.py --num-gpus 4 \
  --config-file configs/coco/instance-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_50ep.yaml