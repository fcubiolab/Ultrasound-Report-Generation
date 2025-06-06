import torch
import torch.nn as nn
import torchvision.models as models

try:
    import timm
except ImportError:  # timm is optional
    timm = None


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained

        if self.visual_extractor.startswith('vit'):
            if timm is None:
                raise ImportError(
                    'timm is required for ViT backbone but is not installed.'
                )
            self.model = timm.create_model(
                self.visual_extractor, pretrained=self.pretrained
            )
            self.out_dim = self.model.embed_dim
            self.avg_fnt = None
        else:
            model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            self.out_dim = 2048
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        self.cov1x1 = nn.Conv2d(in_channels=self.out_dim, out_channels=args.nhidden, kernel_size=(1, 1))
        if self.pretrained is True:
            print('first init the imagenet pretrained!')

    def forward(self, images):
        if self.visual_extractor.startswith('vit'):
            tokens = self.model.forward_features(images)
            avg_feats = tokens[:, 0]
            patch_feats = tokens[:, 1:]
            b, n, c = patch_feats.shape
            h = w = int(n ** 0.5)
            patch_feats_2d = patch_feats.permute(0, 2, 1).reshape(b, c, h, w)
            att_feat_it = self.cov1x1(patch_feats_2d)
            avg_feat_it = att_feat_it.mean(dim=[2, 3])
            return patch_feats, avg_feats, att_feat_it, avg_feat_it
        else:
            patch_feats = self.model(images)
            att_feat_it = self.cov1x1(patch_feats)
            avg_feat_it = self.avg_fnt(att_feat_it).squeeze().reshape(-1, att_feat_it.size(1))
            avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
            batch_size, feat_size, _, _ = patch_feats.shape
            patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            return patch_feats, avg_feats, att_feat_it, avg_feat_it


