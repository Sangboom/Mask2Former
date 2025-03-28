# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmseg.models.builder import BACKBONES
from mask2former.modeling.backbone.ops.modules import MSDeformAttn
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

from mask2former.modeling.backbone.dino_v2 import DinoVisionTransformer
# from mask2former.modeling.meta_arch.cross_view_attention import CrossViewPAnet
from mask2former.modeling.backbone.dino_rein_utils import set_requires_grad, set_train
from mask2former.modeling.backbone.dinov2_layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block

# from .base.vit import TIMMVisionTransformer

from .vitadapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
import fvcore.nn.weight_init as weight_init


_logger = logging.getLogger(__name__)



class ViTAdapterDinoVisionTransformer(DinoVisionTransformer):
    def __init__(self, cfg=None, **kwargs):
        
        pretrain_size=cfg.INPUT.IMAGE_SIZE
        # num_heads=12
        conv_inplane=64
        n_points=4
        deform_num_heads=12 # 12 for base, 16 for large
        init_values=0.
        # interaction_indexes=None
        if cfg.MODEL.DINOV2.SIZE == 'base':
            interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]] # for base
        else:
            interaction_indexes=None # Not implementation
            
        with_cffn=True
        cffn_ratio=0.25
        deform_ratio=0.5 # base
        # deform_ratio=1.0
        add_vit_feature=True
        # pretrained=None,
        use_extra_extractor=True
        with_cp=False

        # super().__init__(num_heads=num_heads, pretrained=pretrained,
        #                  with_cp=with_cp, *args, **kwargs)
        super().__init__(**kwargs)
        
        self.is_adapter = cfg.MODEL.DINOV2.ADAPTER
        self.adapter_type = cfg.MODEL.DINOV2.ADAPTER_TYPE

        self.cfg = cfg
        # self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        # embed_dim = self.embed_dim
        embed_dim = kwargs['embed_dim']
        self.drop_path_rate = 0.3
        drop_rate = 0.
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.cfg = cfg
        self.num_layers = 4
        # self.out_indices = (0, 1, 2, 3)
        if cfg.MODEL.DINOV2.SIZE == 'base':
            self.out_indices = (2, 5, 8, 11)
        elif cfg.MODEL.DINOV2.SIZE == 'large':
            self.out_indices = (5, 11, 17, 23)
        self.xyz_dict = {"res5": 0, "res4": 1, "res3": 2, "res2": 3}
        # self.xyz_dict = {"res5": 1, "res4": 1, "res3": 1, "res2": 1}
        self._out_features = ["res2", "res3", "res4", "res5"]
        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": embed_dim,
            "res3": embed_dim,
            "res4": embed_dim,
            "res5": embed_dim,
        }

        # self.layers = nn.ModuleList()
        # for i_layer in range(self.num_layers):
        #     layer = CrossViewLayer(
        #         dim=int(embed_dim),
        #         cfg=cfg,
        #         layer_i=i_layer
        #     )
        #     self.layers.append(layer)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward_features(self, x, masks=None, x_xyz=None, shape=None, multiview_data=None, decoder_3d=False):
        deform_inputs1, deform_inputs2 = deform_inputs(x)
        
        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        # x, H, W = self.patch_embed(x)
        B, nc, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)

        # Interaction
        outs = list()
        for i, vitadapter_layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = vitadapter_layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())
        
        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        # print("c1 shape", c1.shape)
        # print("c2 shape", c2.shape)
        # print("c2 up shape", self.up(c2).shape)
        c1 = self.up(c2) + c1
        

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        
        outs = {}
        outs["res2"] = f1
        outs["res3"] = f2
        outs["res4"] = f3
        outs["res5"] = f4

        for name, param in self.named_parameters():
            if not param.requires_grad:
                print(f"{name} is frozen!")
            else:
                print(f"{name} is trainable!")
        
        return outs

    ### It's for rein adapter. Need to be refactored.
    def train(self, mode: bool = True):
        if not mode:
            # return super().train(mode)
            # return super(ReinsDinoVisionTransformer, self).train(mode, self.cfg.MODEL.DINOV2.FREEZE_BACKBONE)
            super(ViTAdapterDinoVisionTransformer, self).train(mode, self.cfg.MODEL.DINOV2.FREEZE_BACKBONE)
        
        
        if not self.cfg.MODEL.DINOV2.FREEZE_BACKBONE:
            # for name, param in self.named_parameters():
            #     param.requires_grad = True
            # if self.cfg.MODEL.CROSS_VIEW_BACKBONE:
            #     set_requires_grad(self, ["cls_token", "pos_embed", "patch_embed", "blocks", "layers"])
            #     set_train(self, ["cls_token", "pos_embed", "patch_embed", "blocks", "layers"])
            # else:
            set_requires_grad(self, ["cls_token", "pos_embed", "patch_embed", "blocks"])
            set_train(self, ["cls_token", "pos_embed", "patch_embed", "blocks"])

        else:
            set_requires_grad(self, ["interactions", "level_embed", "spm"])
            set_train(self, ["interactions", "level_embed", "spm"])


class D2DinoViTAdapter(ViTAdapterDinoVisionTransformer, Backbone):
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
