from transformers import Qwen2_5_VLForConditionalGeneration
from custom_cnn_encoder import CustomCNNFeatureEncoder

def patch_qwen2_5_vl_model(model, input_channels, num_visual_tokens):
    """
    动态替换模型的视觉编码器，无需修改源码
    """
    # 创建你的CNN实例
    custom_visual = CustomCNNFeatureEncoder(
        input_channels=input_channels,
        hidden_dim=model.config.hidden_size,
        num_visual_tokens=num_visual_tokens
    ).to(model.device, dtype=model.dtype)
    
    # 替换！（猴子补丁）
    model.visual = custom_visual
    
    # 可选：将原视觉编码器显存释放
    # del model.visual
    # model.visual = custom_visual
    
    return model


def load_custom_model(model_path, input_channels=10, num_visual_tokens=64):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path)
    model = patch_qwen2_5_vl_model(model, input_channels, num_visual_tokens)
    return model