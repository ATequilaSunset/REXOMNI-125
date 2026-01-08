import torch
import torch.nn as nn

class CustomCNNFeatureEncoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, num_visual_tokens):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7)),
        )
        self.proj = nn.Linear(128 * 7 * 7, hidden_dim)
        self.num_visual_tokens = num_visual_tokens

    def forward(self, pixel_values, **kwargs):  # 接收**kwargs忽略grid_thw
        
        features = self.conv(pixel_values)  # [B, 128, 7, 7]
        features = features.flatten(2).transpose(1, 2)  # [B, 49, 128]
        image_embeds = self.proj(features)  # [B, 49, hidden_dim]
        
        # 确保输出token数与num_visual_tokens一致
        if image_embeds.shape[1] != self.num_visual_tokens:
            # 可以通过插值或选择调整
            image_embeds = image_embeds[:, :self.num_visual_tokens, :]
        
        return image_embeds.reshape(-1, image_embeds.shape[-1])  # [B*num_tokens, hidden_dim]
    
