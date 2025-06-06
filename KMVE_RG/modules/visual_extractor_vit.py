import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
import math


class VisualExtractorViT(nn.Module):
    def __init__(self, args):
        super(VisualExtractorViT, self).__init__()
        self.args = args
        
        # 使用預訓練的 ViT 模型
        self.vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        self.hidden_size = self.vit_model.config.hidden_size  # 768 for ViT-base
        
        # 凍結部分層（可選）
        freeze_layers = getattr(args, 'freeze_vit_layers', 0)
        if freeze_layers > 0:
            for i, layer in enumerate(self.vit_model.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # 特徵投影層 - 保持與原始模型相同的維度
        # patch_feats 對應 att_feats (注意力特徵)
        self.patch_proj = nn.Linear(self.hidden_size, args.d_vf)  # 投影到 d_vf (2048)
        
        # avg_feats 對應 fc_feats (全局特徵)
        self.cls_proj = nn.Linear(self.hidden_size, args.d_vf)  # 投影到 d_vf (2048)
        
        # att_feat_it 對應處理後的注意力特徵
        self.att_proj = nn.Linear(self.hidden_size, args.d_model)  # 投影到 d_model (512)
        
        # avg_feat_it 對應處理後的全局特徵 (用於 KMVE)
        self.kmve_proj = nn.Linear(self.hidden_size, args.d_model)  # 投影到 d_model (512)
        
        # Dropout
        self.dropout = nn.Dropout(args.dropout)
        
        # 位置編碼（如果需要額外的位置信息）
        self.use_extra_pos = getattr(args, 'use_extra_pos_encoding', False)
        if self.use_extra_pos:
            self.pos_embedding = nn.Parameter(torch.randn(1, 197, self.hidden_size))  # 197 = 196 patches + 1 CLS
        
        print(f'ViT Visual Extractor initialized with hidden_size: {self.hidden_size}')
        if freeze_layers > 0:
            print(f'Frozen first {freeze_layers} layers of ViT')
    
    def forward(self, images):
        """
        Args:
            images: torch.Tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            patch_feats: 對應原始的 patch_feats, shape (batch_size, num_patches, d_vf)
            avg_feats: 對應原始的 avg_feats, shape (batch_size, d_vf)
            att_feat_it: 對應原始的 att_feat_it, shape (batch_size, d_model, 14, 14)
            avg_feat_it: 對應原始的 avg_feat_it, shape (batch_size, d_model)
        """
        batch_size = images.size(0)
        
        # 確保圖像尺寸為 224x224
        if images.size(-1) != 224 or images.size(-2) != 224:
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # 通過 ViT 提取特徵
        vit_outputs = self.vit_model(images, output_hidden_states=True)
        
        # 獲取最後一層的隱藏狀態
        hidden_states = vit_outputs.last_hidden_state  # (batch_size, 197, 768)
        
        # 額外位置編碼（如果啟用）
        if self.use_extra_pos:
            hidden_states = hidden_states + self.pos_embedding
        
        # 分離 CLS token 和 patch tokens
        cls_token = hidden_states[:, 0]  # (batch_size, 768) - 全局表示
        patch_tokens = hidden_states[:, 1:]  # (batch_size, 196, 768) - 空間特徵
        
        # 1. patch_feats: 對應原始的 patch_feats
        # 原始: (batch_size, feat_size, -1) -> (batch_size, -1, feat_size)
        patch_feats = self.patch_proj(patch_tokens)  # (batch_size, 196, d_vf)
        patch_feats = self.dropout(patch_feats)
        
        # 2. avg_feats: 對應原始的 avg_feats (全局特徵)
        avg_feats = self.cls_proj(cls_token)  # (batch_size, d_vf)
        avg_feats = self.dropout(avg_feats)
        
        # 3. att_feat_it: 對應原始的注意力特徵圖
        # 原始形狀是 (batch_size, d_model, 7, 7)，這裡我們重塑 patch tokens
        att_tokens = self.att_proj(patch_tokens)  # (batch_size, 196, d_model)
        att_tokens = self.dropout(att_tokens)
        
        # 將 196 個 patches 重塑為 14x14 的空間排列 (因為 14*14=196)
        att_feat_it = att_tokens.permute(0, 2, 1).reshape(batch_size, self.args.d_model, 14, 14)
        
        # 4. avg_feat_it: 對應原始的平均特徵 (用於 KMVE)
        avg_feat_it = self.kmve_proj(cls_token)  # (batch_size, d_model)
        avg_feat_it = self.dropout(avg_feat_it)
        
        return patch_feats, avg_feats, att_feat_it, avg_feat_it


class VisualExtractorViTLarge(VisualExtractorViT):
    """使用 ViT-Large 的版本"""
    def __init__(self, args):
        super().__init__(args)
        # 重新初始化為 ViT-Large
        self.vit_model = ViTModel.from_pretrained('google/vit-large-patch16-224')
        self.hidden_size = self.vit_model.config.hidden_size  # 1024 for ViT-large
        
        # 重新定義投影層
        self.patch_proj = nn.Linear(self.hidden_size, args.d_vf)
        self.cls_proj = nn.Linear(self.hidden_size, args.d_vf)
        self.att_proj = nn.Linear(self.hidden_size, args.d_model)
        self.kmve_proj = nn.Linear(self.hidden_size, args.d_model)
        
        print(f'ViT-Large Visual Extractor initialized with hidden_size: {self.hidden_size}')


class VisualExtractorViTCustom(nn.Module):
    """自定義 ViT 配置的版本"""
    def __init__(self, args):
        super(VisualExtractorViTCustom, self).__init__()
        self.args = args
        
        # 創建自定義 ViT 配置
        vit_config = ViTConfig(
            image_size=224,
            patch_size=16,
            num_channels=3,
            hidden_size=getattr(args, 'vit_hidden_size', 768),
            num_hidden_layers=getattr(args, 'vit_num_layers', 12),
            num_attention_heads=getattr(args, 'vit_num_heads', 12),
            intermediate_size=getattr(args, 'vit_intermediate_size', 3072),
            hidden_dropout_prob=args.dropout,
            attention_probs_dropout_prob=args.dropout,
        )
        
        self.vit_model = ViTModel(vit_config)
        self.hidden_size = vit_config.hidden_size
        
        # 特徵投影層
        self.patch_proj = nn.Linear(self.hidden_size, args.d_vf)
        self.cls_proj = nn.Linear(self.hidden_size, args.d_vf)
        self.att_proj = nn.Linear(self.hidden_size, args.d_model)
        self.kmve_proj = nn.Linear(self.hidden_size, args.d_model)
        
        self.dropout = nn.Dropout(args.dropout)
        
        print(f'Custom ViT Visual Extractor initialized with hidden_size: {self.hidden_size}')
    
    def forward(self, images):
        batch_size = images.size(0)
        
        if images.size(-1) != 224 or images.size(-2) != 224:
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        
        vit_outputs = self.vit_model(images)
        hidden_states = vit_outputs.last_hidden_state
        
        cls_token = hidden_states[:, 0]
        patch_tokens = hidden_states[:, 1:]
        
        patch_feats = self.patch_proj(patch_tokens)
        patch_feats = self.dropout(patch_feats)
        
        avg_feats = self.cls_proj(cls_token)
        avg_feats = self.dropout(avg_feats)
        
        att_tokens = self.att_proj(patch_tokens)
        att_tokens = self.dropout(att_tokens)
        att_feat_it = att_tokens.permute(0, 2, 1).reshape(batch_size, self.args.d_model, 14, 14)
        
        avg_feat_it = self.kmve_proj(cls_token)
        avg_feat_it = self.dropout(avg_feat_it)
        
        return patch_feats, avg_feats, att_feat_it, avg_feat_it
