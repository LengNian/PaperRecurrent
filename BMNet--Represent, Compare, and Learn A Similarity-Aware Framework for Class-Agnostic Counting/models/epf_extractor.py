"""
Class agnostic counting
Feature extractors for exemplars.
"""
from torch import nn

class DirectPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, repeat_times=1, use_scale_embedding=True, scale_number=20):
        super().__init__()
        self.repeat_times = repeat_times # 1
        self.use_scale_embedding = use_scale_embedding

        self.patch2query = nn.Linear(input_dim, hidden_dim) # align the patch feature dim to query patch dim.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # pooling used for the query patch feature
        self._weight_init_()
        if self.use_scale_embedding:
            # 这里的scale_number就是尺度等级(嵌入层的词汇表大小)
            # 嵌入层的作用就是将离散的整数索引映射为连续的向量表示
            self.scale_embedding = nn.Embedding(scale_number, hidden_dim)

    # 这里传入的是patch_feature和scale_embedding
    def forward(self, patch_feature, scale_index):
        bs, batch_num_patches = scale_index.shape
        # patch_feature: [24, 1024, 8, 8]
        # avg --> flatten: [24, 1024, 8, 8] --> [24, 1024, 1, 1] --> [24, 1024]
        patch_feature = self.avgpool(patch_feature).flatten(1) # bs X patchnumber X feature_dim

        # [24, 256] --> [24, 256] --> [8, 3, 256] --> [3, 8, 256]
        patch_feature = self.patch2query(patch_feature) \
            .view(bs, batch_num_patches, -1) \
            .repeat_interleave(self.repeat_times, dim=1) \
            .permute(1, 0, 2) \
            .contiguous() 
        
        if self.use_scale_embedding:
            # [8, 3] --> [8, 3, 256]
            scale_embedding = self.scale_embedding(scale_index) # bs X number_query X dim
            # [8, 3, 256]
            patch_feature = patch_feature + scale_embedding.permute(1, 0, 2)
        
        return patch_feature
    
    def _weight_init_(self):
        for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_uniform_(
                #         m.weight, 
                #         mode='fan_in', 
                #         nonlinearity='relu'
                #         )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def build_epf_extractor(cfg):
    extractor_name = cfg.MODEL.epf_extractor
    input_dim = 1024 if cfg.MODEL.backbone_layer == 'layer3' else 2048
    if extractor_name == 'direct_pooling':
        return DirectPooling(input_dim=input_dim,
                             hidden_dim=cfg.MODEL.hidden_dim,
                             repeat_times=cfg.MODEL.repeat_times,
                             use_scale_embedding=cfg.MODEL.ep_scale_embedding,
                             scale_number=cfg.MODEL.ep_scale_number)
    else:
        raise NotImplementedError
        