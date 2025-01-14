"""
Basic class agnostic counting model with backbone, refiner, matcher and counter.
"""
import torch
from torch import nn

class CACModel(nn.Module):
    """ Class Agnostic Counting Model"""
    
    def __init__(self, backbone, EPF_extractor, refiner, matcher, counter, hidden_dim):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            EPF_extractor: torch module of the feature extractor for patches. See epf_extractor.py
            repeat_times: Times to repeat each exemplar in the transformer decoder, i.e., the features of exemplar patches.
        """
        super().__init__()
        self.EPF_extractor = EPF_extractor
        self.refiner = refiner
        self.matcher = matcher
        self.counter = counter

        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        
    def forward(self, samples: torch.Tensor, patches: torch.Tensor, is_train: bool):
        """ The forward expects samples containing query images and corresponding exemplar patches.
            samples is a stack of query images, of shape [batch_size X 3 X H X W]
            patches is a torch Tensor, of shape [batch_size x num_patches x 3 x 128 x 128]
            The size of patches are small than samples

            It returns a dict with the following elements:
               - "density_map": Shape= [batch_size x 1 X h_query X w_query]
               - "patch_feature": Features vectors for exemplars, not available during testing.
                                  They are used to compute similarity loss. 
                                Shape= [exemplar_number x bs X hidden_dim]
               - "img_feature": Feature maps for query images, not available during testing.
                                Shape= [batch_size x hidden_dim X h_query X w_query]
            
        """
        # Stage 1: extract features for query images and exemplars
        scale_embedding, patches = patches['scale_embedding'], patches['patches']
        # samples: [8, 3, w, h] 这里的samples其实就是query可以看engine.py中 outputs = model(img, patches, is_train=True), 前向传播传入的是engine.py
        # patches: [8, 3, 3, 128, 128]
        # 这里的samples就是backbone类中继承的Backbonebase中forward的tensorlist
        # features: [8, 3, 384, 576]  -->  [8, 1024, 24, 36]  这里以384, 576为例, 这里通道变为1024和论文对应
        features = self.backbone(samples)
        # [8, 1024, 24, 36]  -->  [8, 256, 24, 36]
        features = self.input_proj(features)

        # patches: [8, 3, 3, 128, 128] --> [24, 3, 128, 128]
        patches = patches.flatten(0, 1)

        # patch_feature: [24, 1024, 8, 8]  ([24, 3, 128, 128] --> [24, 1024, 8, 8])
        patch_feature = self.backbone(patches) # obtain feature maps for exemplar patches

        # patch_feature: [24, 1024, 8, 8] --> [3, 8, 256]  # 对应论文中将exemplar压缩为256D vector
        # scale_embedding: [8, 3]
        # 压缩exemplar同时把scale_embedding信息嵌入
        patch_feature = self.EPF_extractor(patch_feature, scale_embedding) # compress the feature maps into vectors and inject scale embeddings
        
        # Stage 2: enhance feature representation, e.g., the self similarity module.

        # features --> refined_feature: [8, 256, 24, 36] --> [8, 256, 24, 36]
        # patch_feature --> patch_feature: [3, 8, 256] --> [3, 8, 256]
        refined_feature, patch_feature = self.refiner(features, patch_feature)

        # Stage 3: generate similarity map by densely measuring similarity.
        # refined_feature --> counting_feature: [8, 256, 24, 36] --> [8, 257, 24, 36]
        # counting_feature: 就是自注意力之后融合过的特征和相似图堆叠的特征图
        # corr_map: [8, 864, 3]  相似图(是没有平均过的相似图)
        counting_feature, corr_map = self.matcher(refined_feature, patch_feature)

        # Stage 4: predicting density map
        # density_map: [8, 1, 384, 576]
        density_map = self.counter(counting_feature)
        
        if not is_train:
            return density_map
        else:
            return {'corr_map': corr_map, 'density_map': density_map}

    #def _reset_parameters(self):
    #    for p in self.parameters():
    #        if p.dim() > 1:
    #            nn.init.x, avier_uniform_(p)
