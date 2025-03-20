

from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable
import numpy as np

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

from odin.modeling.backbone.dinov2_layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block
from odin.modeling.backbone.dino_v2 import DinoVisionTransformer
from odin.modeling.backbone.reins import Reins
from odin.modeling.backbone.dino_rein_utils import set_requires_grad, set_train
from odin.modeling.meta_arch.cross_view_attention import CrossViewPAnet

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from detectron2.modeling.backbone.fpn import _assert_strides_are_log2_contiguous




class CrossViewLayer(nn.Module):
    """Cross View Attention Layer which translate to 3D and apply cross view attention.
    Args:
        dim (int): Number of feature channels
    """

    def __init__(
        self,
        dim,
        cfg=None,
        layer_i=None
    ):
        super().__init__()
        
        self.cfg = cfg

        if cfg.MODEL.CROSS_VIEW_BACKBONE:
            # we don't find much benefit doing cross view attention in the first layer
            # and it's expensive, so we only do it in the later layers
            if layer_i == 0:
                self.cross_view_attention = False
            else:
                self.cross_view_attention = True
                conv_dim = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
                self.res_to_trans = nn.Sequential(
                        nn.Conv2d(dim, conv_dim, kernel_size=1),
                        nn.GroupNorm(32, conv_dim),
                    )
                weight_init.c2_xavier_fill(self.res_to_trans[0])
                self.cross_view_attn = CrossViewPAnet(
                        latent_dim=conv_dim, nsample=self.cfg.MODEL.KNN,
                        dropout=self.cfg.MODEL.MASK_FORMER.DROPOUT, num_layers=self.cfg.MODEL.CROSS_VIEW_NUM_LAYERS[layer_i],
                        cfg=cfg
                    )
                self.trans_to_res = nn.Sequential(
                        nn.Conv2d(conv_dim, dim, kernel_size=1),
                        nn.GroupNorm(32, dim),
                    )
                weight_init.c2_xavier_fill(self.trans_to_res[0])
                # self.cross_layer_norm = nn.LayerNorm(dim)


    def forward(self, x, H, W, x_xyz=None, shape=None, multiview_data=None, decoder_3d=False):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        if decoder_3d and self.cfg.MODEL.CROSS_VIEW_BACKBONE and self.cross_view_attention:

            # project x
            x2 = x.view(x.shape[0], H, W, -1).permute(0, 3, 1, 2).contiguous()
            x2 = self.res_to_trans(x2)

            # cross view attention
            x2 = self.cross_view_attn(
                feature_list=[x2],
                xyz_list=x_xyz,
                shape=shape[:2],
                multiview_data=multiview_data,
                voxelize=self.cfg.INPUT.VOXELIZE
            )[0]

            # project back
            x2 = self.trans_to_res(x2)
            x2 = x2.permute(0, 2, 3, 1).flatten(1, 2).contiguous()
            
            # skip connection
            x = x + x2
        
        return x, H, W



class SimpleFeaturePyramid(nn.Module):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        net,
        in_feature,
        out_channels,
        scale_factors,
        top_block=None,
        norm="LN",
        square_pad=0,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            in_feature (str): names of the input feature maps coming
                from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm (str): the normalization to use.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(SimpleFeaturePyramid, self).__init__()
        assert isinstance(net, Backbone)

        self.scale_factors = scale_factors

        input_shapes = net.output_shape()
        strides = [int(input_shapes[in_feature].stride / scale) for scale in scale_factors]
        _assert_strides_are_log2_contiguous(strides)

        dim = input_shapes[in_feature].channels
        self.stages = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                    Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        self.net = net
        self.in_feature = in_feature
        self.top_block = top_block
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        # self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        self._out_feature_strides = {"res{}".format(int(math.log2(s))-1): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                # self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)
                self._out_feature_strides["res{}".format(s)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad

    @property
    def padding_constraints(self):
        return {
            "size_divisiblity": self._size_divisibility,
            "square_size": self._square_pad,
        }

    def forward(self, x, x_xyz=None, shape=None, multiview_data=None, decoder_3d=False):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = self.net(x, x_xyz, shape, multiview_data, decoder_3d)
        features = bottom_up_features[self.in_feature]
        results = []

        for stage in self.stages:
            results.append(stage(features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

class ReinsDinoVisionTransformer(DinoVisionTransformer):
    def __init__(
        self,
        cfg,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.is_adapter = cfg.MODEL.DINOV2.ADAPTER
        self.adapter_type = cfg.MODEL.DINOV2.ADAPTER_TYPE
        
        if self.is_adapter:
            if self.adapter_type == 'rein':
                self.reins = Reins(
                    num_layers = kwargs['depth'],
                    embed_dims = kwargs['embed_dim'],
                    patch_size = kwargs['patch_size'],
                )
        
        # embed_dim = cfg.MODEL.SWIN.EMBED_DIM
        embed_dim = kwargs['embed_dim']

        self.simple_fpn = cfg.MODEL.SIMPLE_FPN

        self.cfg = cfg
        self.num_layers = 4
        # self.out_indices = (0, 1, 2, 3)
        if cfg.MODEL.DINOV2.SIZE == 'base':
            self.out_indices = (2, 5, 8, 11)
        elif cfg.MODEL.DINOV2.SIZE == 'large':
            self.out_indices = (5, 11, 17, 23)
        self.xyz_dict = {"res5": 1, "res4": 1, "res3": 1, "res2": 1}
        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_strides = {
            "res2": 32,
            "res3": 32,
            "res4": 32,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": embed_dim,
            "res3": embed_dim,
            "res4": embed_dim,
            "res5": embed_dim,
        }

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = CrossViewLayer(
                dim=int(embed_dim),
                cfg=cfg,
                layer_i=i_layer
            )
            self.layers.append(layer)
        

    def forward_features(self, x, masks=None, x_xyz=None, shape=None, multiview_data=None, decoder_3d=False):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = {}
        # outs = []

        for idx, blk in enumerate(self.blocks):
            print(f"Before block {idx}: {x.requires_grad}")
            x = blk(x)
            print(f"After block {idx}: {x.requires_grad}")
            if self.is_adapter:
                if self.adapter_type == 'rein':
                    x = self.reins.forward(
                        x,
                        idx,
                        batch_first=True,
                        has_cls_token=True,
                    )
            
            if idx in self.out_indices:
                if self.cfg.MODEL.CROSS_VIEW_BACKBONE and decoder_3d:
                    mv_data = {}
                    if self.cfg.MODEL.DINOV2.SIZE == 'base':
                        mv_data['multi_scale_p2v'] = [multiview_data['multi_scale_p2v'][self.xyz_dict[f"res{int((idx+1)/3 + 1)}"]]]
                    elif self.cfg.MODEL.DINOV2.SIZE == 'large':
                        mv_data['multi_scale_p2v'] = [multiview_data['multi_scale_p2v'][self.xyz_dict[f"res{int((idx+1)/6 + 1)}"]]]

                    # mv_data['multi_scale_p2v'] = [multiview_data['multi_scale_p2v'][self.xyz_dict[f"res{idx+2}"]]]
                    # mv_data['multi_scale_p2v'] = [multiview_data['multi_scale_p2v'][1]]
                    
                    if self.cfg.MODEL.DINOV2.SIZE == 'base':
                        xyz = [x_xyz[self.xyz_dict[f"res{int((idx+1)/3 + 1)}"]]]
                        # xyz = [x_xyz[self.xyz_dict[f"res{idx+2}"]]]
                        # xyz = [x_xyz[1]]
                        layer = self.layers[int((idx+1)/3 - 1)]
                        # layer = self.layers[idx]
                    elif self.cfg.MODEL.DINOV2.SIZE == 'large':
                        xyz = [x_xyz[self.xyz_dict[f"res{int((idx+1)/6 + 1)}"]]]
                        layer = self.layers[int((idx+1)/6 - 1)]

                    x_cls = x[:, 0:1, :]
                    x_reg = x[:, 1:, :]

                    x_out, H, W = layer(
                        x_reg, H, W, 
                        x_xyz=xyz, shape=shape, multiview_data=mv_data, 
                        decoder_3d=decoder_3d)
                    x = torch.cat([x_cls, x_out], dim=1)
                else:
                    mv_data = None
                    xyz = None
            else:
                mv_data = None
                xyz = None
            
            
            if idx in self.out_indices:
                # norm_layer = getattr(self, f"norm{idx}")
                # x_out = norm_layer(x_out)

                # x_out = x[:, 1:, :]
                # x_out = x[:, 1:, :]
                # x_out[:, 0, :] = x_out[:, 0, :] + x[:, 0, :]
                if self.cfg.MODEL.CROSS_VIEW_BACKBONE:
                    out = x_out.view(-1, H, W, self.num_features).permute(0, 3, 1, 2).contiguous()
                else:
                    out = x[:, 1:, :].view(-1, H, W, self.num_features).permute(0, 3, 1, 2).contiguous()
                # out = x_out.view(-1, H, W, self.num_features[idx]).permute(0, 3, 1, 2).contiguous()

                if self.cfg.MODEL.DINOV2.SIZE == 'base':
                    outs["res{}".format(int((idx+1)/3 + 1))] = out
                elif self.cfg.MODEL.DINOV2.SIZE == 'large':
                    outs["res{}".format(int((idx+1)/6 + 1))] = out
        
        if self.simple_fpn == False:
            outs["res2"] = F.interpolate(outs["res2"], scale_factor=4, mode="bilinear", align_corners=False)
            outs["res3"] = F.interpolate(outs["res3"], scale_factor=2, mode="bilinear", align_corners=False)
            outs["res5"] = F.interpolate(outs["res5"], scale_factor=0.5, mode="bilinear", align_corners=False)
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                print(f"{name} is frozen!")
            else:
                print(f"{name} is trainable!")

        return outs

    def forward_features_full_rein(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.reins.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
            )
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
        return self.reins.return_auto(outs)

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            # return self.head(ret["x_norm_clstoken"])
            return ret
        

    def forward_features_no_rein(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
        return x

    def train(self, mode: bool = True):
        if not mode:
            # return super().train(mode)
            # return super(ReinsDinoVisionTransformer, self).train(mode, self.cfg.MODEL.DINOV2.FREEZE_BACKBONE)
            super(ReinsDinoVisionTransformer, self).train(mode, self.cfg.MODEL.DINOV2.FREEZE_BACKBONE)
        
        
        if not self.cfg.MODEL.DINOV2.FREEZE_BACKBONE:
            # for name, param in self.named_parameters():
            #     param.requires_grad = True
            if self.cfg.MODEL.CROSS_VIEW_BACKBONE:
                set_requires_grad(self, ["cls_token", "pos_embed", "patch_embed", "blocks", "layers"])
                set_train(self, ["cls_token", "pos_embed", "patch_embed", "blocks", "layers"])
            else:
                set_requires_grad(self, ["cls_token", "pos_embed", "patch_embed", "blocks"])
                set_train(self, ["cls_token", "pos_embed", "patch_embed", "blocks"])

        else:
            if self.is_adapter and self.adapter_type == 'rein':
                # Setting Rein Train
                if self.cfg.MODEL.CROSS_VIEW_BACKBONE:
                    # set_requires_grad(self, ["reins", "layers"])
                    # set_train(self, ["reins", "layers"])
                    set_requires_grad(self, ["reins"])
                    set_train(self, ["reins"])
                    set_requires_grad(self, ["layers"])
                    set_train(self, ["layers"])
                else:
                    set_requires_grad(self, ["reins"])
                    set_train(self, ["reins"])
            
            # if self.cfg.MODEL.CROSS_VIEW_BACKBONE:
            #     # Setting Crossview attn 
            #     set_requires_grad(self, ['layers'])
            #     set_train(self, ['layers'])
            #     # set_requires_grad(self, ['cross_view_attn', 'cross_layer_norm', 'res_to_trans', 'trans_to_res'])
            #     # set_train(self, ['cross_view_attn', 'cross_layer_norm', 'res_to_trans', 'trans_to_res'])

        # for name, param in self.named_parameters():
        #     if not param.requires_grad:
        #         print(f"{name} is frozen! train!")
        #     else:
        #         print(f"{name} is trainable! train!")
        
        # exit()


class D2SDinoVisionTransformer(ReinsDinoVisionTransformer, Backbone):
    def __init__(self, cfg, input_shape):
        patch_size=16
        num_register_tokens=0
        img_size = cfg.INPUT.IMAGE_SIZE

        if cfg.MODEL.DINOV2.SIZE == 'base':
            embed_dim = 768
            depth = 12
            num_heads = 12
            mlp_ratio = 4
        elif cfg.MODEL.DINOV2.SIZE == 'large':
            embed_dim = 1024
            depth = 24
            num_heads = 16
            mlp_ratio = 4
        else:
            embed_dim = 768
            depth = 12
            num_heads = 12
            mlp_ratio = 4

        super().__init__(
            cfg=cfg,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            block_chunks=0,
            block_fn=partial(Block, attn_class=MemEffAttention),
            num_register_tokens=num_register_tokens,
            img_size=img_size
        )
        # self.cfg = cfg

        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
            # "res2": 32,
            # "res3": 32,
            # "res4": 32,
            # "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.embed_dim,
            "res3": self.embed_dim,
            "res4": self.embed_dim,
            "res5": self.embed_dim,
        }

    def forward(self, x, x_xyz=None, shape=None, multiview_data=None, decoder_3d=False):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        # outputs = {}
        outputs = super().forward(x, x_xyz=x_xyz, shape=shape, multiview_data=multiview_data, decoder_3d=decoder_3d)
        # for k in y.keys():
        #     if k in self._out_features:
        #         outputs[k] = y[k]
        
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32