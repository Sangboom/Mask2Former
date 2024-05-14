export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# python train_net.py --num-gpus 8 \
#   --config-file configs/coco/instance-segmentation/maskformer2_R50_bs16_1x_ms.yaml --resume --eval-only

python train_net.py --num-gpus 8 \
  --config-file configs/cityscapes/instance-segmentation/maskformer2_R50_bs8_27k.yaml