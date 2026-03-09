"""
改进的12类别多模态模型架构
核心改进：
1. 使用ResNet18替代SimpleCNN
2. Co-Attention融合机制
3. 辅助分类头（aux heads）
4. 多尺度特征融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ============================================================================
# 改进的Audio Backbone（Mel-Spectrogram CNN）
# ============================================================================
class ImprovedAudioBackbone(nn.Module):
    """
    基于Mel频谱图的音频编码器
    使用类似ResNet的结构
    """
    def __init__(self, input_channels=1, output_dim=128):
        super().__init__()
        
        # 初始卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # ResBlock 1
        self.res_block1 = self._make_res_block(32, 64, stride=2)
        
        # ResBlock 2
        self.res_block2 = self._make_res_block(64, 128, stride=2)
        
        # ResBlock 3
        self.res_block3 = self._make_res_block(128, 256, stride=2)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
    
    def _make_res_block(self, in_channels, out_channels, stride=1):
        """创建残差块"""
        layers = []
        
        # 第一个卷积
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # 第二个卷积
        layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        
        # 残差连接
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W] Mel-spectrogram
        Returns:
            features: [B, output_dim]
        """
        # x shape: [B, 1, H, W]
        x = self.conv1(x)  # [B, 32, H, W]
        
        # ResBlocks
        identity = x
        x = self.res_block1(x)  # [B, 64, H/2, W/2]
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = x + identity
        x = F.relu(x)
        
        x = self.res_block2(x)  # [B, 128, H/4, W/4]
        x = self.res_block3(x)  # [B, 256, H/8, W/8]
        
        # 全局池化
        x = self.global_pool(x)  # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 256]
        
        # 投影
        features = self.projection(x)  # [B, output_dim]
        
        return features


# ============================================================================
# 改进的Video Backbone（2D ResNet）
# ============================================================================
class ImprovedVideoBackbone(nn.Module):
    """
    基于ResNet18的视频编码器
    使用预训练权重
    """
    def __init__(self, output_dim=512, pretrained=True):
        super().__init__()
        
        # 加载预训练ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W] 视频帧
        Returns:
            features: [B, output_dim]
        """
        B, T, C, H, W = x.shape
        
        # 处理每一帧
        x = x.view(B * T, C, H, W)  # [B*T, C, H, W]
        
        # 提取特征
        x = self.features(x)  # [B*T, 512, 1, 1]
        x = x.view(x.size(0), -1)  # [B*T, 512]
        
        # 时间维度聚合（平均池化）
        x = x.view(B, T, -1)  # [B, T, 512]
        x = x.mean(dim=1)  # [B, 512]
        
        # 投影
        features = self.projection(x)  # [B, output_dim]
        
        return features


# ============================================================================
# Co-Attention融合模块
# ============================================================================
class CoAttentionFusion(nn.Module):
    """
    Co-Attention机制：让音视频特征互相关注
    """
    def __init__(self, audio_dim=128, video_dim=512, d_model=256, num_heads=8, dropout=0.2):
        super().__init__()
        
        # 投影到相同维度
        self.audio_proj = nn.Linear(audio_dim, d_model)
        self.video_proj = nn.Linear(video_dim, d_model)
        
        # Multi-head attention
        self.audio_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.video_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Layer Norm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, audio_feat, video_feat):
        """
        Args:
            audio_feat: [B, audio_dim]
            video_feat: [B, video_dim]
        Returns:
            fused_feat: [B, d_model]
        """
        # 投影
        audio = self.audio_proj(audio_feat).unsqueeze(1)  # [B, 1, d_model]
        video = self.video_proj(video_feat).unsqueeze(1)  # [B, 1, d_model]
        
        # Audio attends to Video
        audio_attended, _ = self.audio_attn(audio, video, video)  # [B, 1, d_model]
        audio_attended = self.ln1(audio_attended.squeeze(1))
        
        # Video attends to Audio
        video_attended, _ = self.video_attn(video, audio, audio)  # [B, 1, d_model]
        video_attended = self.ln2(video_attended.squeeze(1))
        
        # 拼接
        combined = torch.cat([audio_attended, video_attended], dim=1)  # [B, d_model*2]
        
        # Feed-forward
        fused = self.ff(combined)  # [B, d_model]
        
        return fused


# ============================================================================
# 完整的多模态模型
# ============================================================================
class ImprovedMultimodalModel(nn.Module):
    """
    改进的12类别多模态分类模型
    """
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Backbones
        self.audio_backbone = ImprovedAudioBackbone(
            input_channels=1,
            output_dim=config.model.audio_dim
        )
        
        self.video_backbone = ImprovedVideoBackbone(
            output_dim=config.model.video_dim,
            pretrained=config.model.pretrained
        )
        
        # 融合模块
        if config.fusion.type == 'coattention':
            self.fusion = CoAttentionFusion(
                audio_dim=config.model.audio_dim,
                video_dim=config.model.video_dim,
                d_model=config.fusion.d_model,
                num_heads=config.fusion.num_heads,
                dropout=config.fusion.dropout
            )
            fusion_dim = config.fusion.d_model
        else:
            # 简单拼接
            self.fusion = None
            fusion_dim = config.model.audio_dim + config.model.video_dim
        
        # 主分类头
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, config.model.num_classes)
        )
        
        # 辅助分类头（用于多任务学习）
        if config.model.use_aux_heads:
            self.aux_audio_classifier = nn.Linear(config.model.audio_dim, config.model.num_classes)
            self.aux_video_classifier = nn.Linear(config.model.video_dim, config.model.num_classes)
        else:
            self.aux_audio_classifier = None
            self.aux_video_classifier = None
    
    def forward(self, audio, video, return_features=False):
        """
        Args:
            audio: [B, 1, H, W] Mel-spectrogram
            video: [B, T, C, H, W] 视频帧
            return_features: 是否返回中间特征
        Returns:
            logits: [B, num_classes]
            aux_outputs: dict (如果使用辅助头)
        """
        # 提取特征
        audio_feat = self.audio_backbone(audio)  # [B, audio_dim]
        video_feat = self.video_backbone(video)  # [B, video_dim]
        
        # 融合
        if self.fusion is not None:
            fused_feat = self.fusion(audio_feat, video_feat)  # [B, d_model]
        else:
            fused_feat = torch.cat([audio_feat, video_feat], dim=1)
        
        # 主分类
        logits = self.classifier(fused_feat)  # [B, num_classes]
        
        # 辅助输出
        aux_outputs = {}
        if self.aux_audio_classifier is not None:
            aux_outputs['audio_logits'] = self.aux_audio_classifier(audio_feat)
            aux_outputs['video_logits'] = self.aux_video_classifier(video_feat)
        
        if return_features:
            aux_outputs['audio_feat'] = audio_feat
            aux_outputs['video_feat'] = video_feat
            aux_outputs['fused_feat'] = fused_feat
        
        return logits, aux_outputs


# ============================================================================
# 模型工厂函数
# ============================================================================
def build_model(config):
    """
    根据配置构建模型
    """
    model = ImprovedMultimodalModel(config)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
    
    return model


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    from types import SimpleNamespace
    
    # 模拟配置
    config = SimpleNamespace(
        model=SimpleNamespace(
            audio_dim=128,
            video_dim=512,
            num_classes=12,
            pretrained=True,
            use_aux_heads=True
        ),
        fusion=SimpleNamespace(
            type='coattention',
            d_model=256,
            num_heads=8,
            dropout=0.2
        )
    )
    
    # 构建模型
    model = build_model(config)
    
    # 测试前向传播
    batch_size = 4
    audio = torch.randn(batch_size, 1, 128, 128)  # Mel-spectrogram
    video = torch.randn(batch_size, 16, 3, 224, 224)  # 16帧视频
    
    logits, aux_outputs = model(audio, video, return_features=True)
    
    print(f"\n输入形状:")
    print(f"  Audio: {audio.shape}")
    print(f"  Video: {video.shape}")
    print(f"\n输出形状:")
    print(f"  Logits: {logits.shape}")
    print(f"  Audio features: {aux_outputs['audio_feat'].shape}")
    print(f"  Video features: {aux_outputs['video_feat'].shape}")
    print(f"  Fused features: {aux_outputs['fused_feat'].shape}")
    if 'audio_logits' in aux_outputs:
        print(f"  Aux audio logits: {aux_outputs['audio_logits'].shape}")
        print(f"  Aux video logits: {aux_outputs['video_logits'].shape}")
    
    print("\n✓ 模型测试通过！")
