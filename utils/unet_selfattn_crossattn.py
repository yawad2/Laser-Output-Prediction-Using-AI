import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def downsample(input_channels, output_channels, kernel_size, apply_batchnorm=True, dropout_prob=0.0, weight_mean=0, weight_sd=0.02):
    layers = [nn.Conv2d(input_channels, output_channels, kernel_size, stride=2, padding=1, bias=False)]

    # Initialize the weights with mean and standard deviation
    nn.init.normal_(layers[0].weight, mean=weight_mean, std=weight_sd)

    if apply_batchnorm:
        layers.append(nn.BatchNorm2d(output_channels))
    layers.append(nn.LeakyReLU(0.2))

    if dropout_prob > 0.0:
        layers.append(nn.Dropout(dropout_prob))

    return nn.Sequential(*layers)

def upsample(input_channels, output_channels, kernel_size, apply_batchnorm=True, dropout_prob=0.0, weight_mean=0, weight_sd=0.02):
    layers = [nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride=2, padding=1, bias=False)]

    # Initialize the weights with mean and standard deviation
    nn.init.normal_(layers[0].weight, mean=weight_mean, std=weight_sd)

    if apply_batchnorm:
        layers.append(nn.BatchNorm2d(output_channels))
    layers.append(nn.ReLU())

    if dropout_prob > 0.0:
        layers.append(nn.Dropout(dropout_prob))

    return nn.Sequential(*layers)
    
class FullSelfAttention(nn.Module):

    def __init__(self, in_channels, num_heads=8, dropout=0.1):
        super(FullSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        assert in_channels % num_heads == 0, 
        
        # Multi-head projections
        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
        # Output projection
        self.out_proj = nn.Conv2d(in_channels, in_channels, 1)
        
        # Learnable scaling and normalization
        self.gamma = nn.Parameter(torch.zeros(1))
        self.norm = nn.GroupNorm(min(32, in_channels), in_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Positional encoding for better spatial understanding
        self.register_buffer('pos_encoding', None)
        
    def get_positional_encoding(self, H, W, device):

        if self.pos_encoding is None or self.pos_encoding.shape[-2:] != (H, W):
            pe = torch.zeros(self.in_channels, H, W, device=device)
            
            # Generate position indices
            y_pos = torch.arange(H, device=device).float().unsqueeze(1).repeat(1, W)
            x_pos = torch.arange(W, device=device).float().unsqueeze(0).repeat(H, 1)
            
            # Normalize positions
            y_pos = y_pos / H
            x_pos = x_pos / W
            
            # Apply sinusoidal encoding
            for i in range(0, self.in_channels // 4):
                div_term = torch.exp(torch.arange(0, 4, 2, device=device).float() * 
                                   -(math.log(10000.0) / 4))
                
                pe[i*4] = torch.sin(y_pos * div_term[0])
                pe[i*4+1] = torch.cos(y_pos * div_term[0])
                pe[i*4+2] = torch.sin(x_pos * div_term[1])
                pe[i*4+3] = torch.cos(x_pos * div_term[1])
            
            self.pos_encoding = pe.unsqueeze(0)
        
        return self.pos_encoding
        
    def forward(self, x):
        B, C, H, W = x.size()
        
        # Add positional encoding
        pos_enc = self.get_positional_encoding(H, W, x.device)
        x_with_pos = x + pos_enc
        
        # Normalize
        x_norm = self.norm(x_with_pos)
        
        # Generate Q, K, V
        Q = self.query(x_norm).view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        K = self.key(x_norm).view(B, self.num_heads, self.head_dim, H * W)
        V = self.value(x_norm).view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attention = torch.matmul(Q, K) / scale  # (B, heads, H*W, H*W)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, V)  # (B, heads, H*W, head_dim)
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        
        # Output projection
        out = self.out_proj(out)
        
        # Residual connection with learnable scaling
        return x + self.gamma * out

class HierarchicalAttention(nn.Module):
    """Multi-scale attention that processes at different resolutions"""
    def __init__(self, in_channels, scales=[1, 2, 4], num_heads=8):
        super(HierarchicalAttention, self).__init__()
        self.scales = scales
        self.attentions = nn.ModuleList([
            FullSelfAttention(in_channels, num_heads) for _ in scales
        ])
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * len(scales), in_channels, 3, padding=1),
            nn.GroupNorm(min(32, in_channels), in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1)
        )
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        B, C, H, W = x.size()
        multi_scale_features = []
        
        for scale, attention in zip(self.scales, self.attentions):
            if scale > 1:
                # Downsample
                x_scaled = F.avg_pool2d(x, scale, scale)
                # Apply attention
                attn_out = attention(x_scaled)
                # Upsample back
                attn_out = F.interpolate(attn_out, size=(H, W), mode='bilinear', align_corners=False)
            else:
                attn_out = attention(x)
            
            multi_scale_features.append(attn_out)
        
        # Fuse multi-scale features
        fused = torch.cat(multi_scale_features, dim=1)
        fused = self.fusion(fused)
        
        return x + self.gamma * fused

class CrossScaleAttention(nn.Module):
    """Cross-attention between different resolution features"""
    def __init__(self, high_res_channels, low_res_channels, num_heads=8):
        super(CrossScaleAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = high_res_channels // num_heads
        
        # Project low-res features to match high-res
        self.low_res_proj = nn.Conv2d(low_res_channels, high_res_channels, 1)
        
        # Multi-head projections
        self.query = nn.Conv2d(high_res_channels, high_res_channels, 1)  # From high-res
        self.key = nn.Conv2d(high_res_channels, high_res_channels, 1)    # From low-res
        self.value = nn.Conv2d(high_res_channels, high_res_channels, 1)  # From low-res
        
        self.out_proj = nn.Conv2d(high_res_channels, high_res_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Normalization
        self.norm_high = nn.GroupNorm(min(32, high_res_channels), high_res_channels)
        self.norm_low = nn.GroupNorm(min(32, high_res_channels), high_res_channels)
        
    def forward(self, high_res_features, low_res_features):
        B, C_h, H_h, W_h = high_res_features.size()
        
        # Project and resize low-res features
        low_res_proj = self.low_res_proj(low_res_features)
        low_res_resized = F.interpolate(low_res_proj, size=(H_h, W_h), mode='bilinear', align_corners=False)
        
        # Normalize
        high_norm = self.norm_high(high_res_features)
        low_norm = self.norm_low(low_res_resized)
        
        # Generate Q from high-res, K,V from low-res
        Q = self.query(high_norm).view(B, self.num_heads, self.head_dim, H_h * W_h).transpose(2, 3)
        K = self.key(low_norm).view(B, self.num_heads, self.head_dim, H_h * W_h)
        V = self.value(low_norm).view(B, self.num_heads, self.head_dim, H_h * W_h).transpose(2, 3)
        
        # Cross-attention
        scale = math.sqrt(self.head_dim)
        attention = torch.matmul(Q, K) / scale
        attention = F.softmax(attention, dim=-1)
        
        out = torch.matmul(attention, V)
        out = out.transpose(2, 3).contiguous().view(B, C_h, H_h, W_h)
        out = self.out_proj(out)
        
        return high_res_features + self.gamma * out

class AttentionResidualBlock(nn.Module):
    """Residual block with attention"""
    def __init__(self, channels, num_heads=8):
        super(AttentionResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(32, channels), channels)
        self.norm2 = nn.GroupNorm(min(32, channels), channels)
        
        self.attention = FullSelfAttention(channels, num_heads)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        
        # First conv block
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        # Add residual
        out = out + residual
        
        # Apply attention
        out = self.attention(out)
        
        return self.relu(out)

class PowerfulGenerator(nn.Module):

    def __init__(self, 
                 input_channels=1, 
                 output_channels=1,
                 use_hierarchical_attention=True,
                 use_cross_scale_attention=True,
                 use_residual_blocks=True,
                 num_heads=16,
                 dropout_prob=0.1):
        super(PowerfulGenerator, self).__init__()
        
        self.use_hierarchical_attention = use_hierarchical_attention
        self.use_cross_scale_attention = use_cross_scale_attention
        self.use_residual_blocks = use_residual_blocks
        
        # original downsample layers
        self.conv_layers = nn.ModuleList([
            downsample(input_channels, 64, 4, dropout_prob=dropout_prob),
            downsample(64, 128, 4, dropout_prob=dropout_prob),
            downsample(128, 256, 4, dropout_prob=dropout_prob),
            downsample(256, 512, 4, dropout_prob=dropout_prob),
            downsample(512, 512, 4, dropout_prob=dropout_prob),
            downsample(512, 512, 4, dropout_prob=dropout_prob),
            downsample(512, 512, 4, dropout_prob=dropout_prob),
            downsample(512, 512, 4, dropout_prob=dropout_prob)
        ])
        
        # original upsample layers 
        self.up_layers = nn.ModuleList([
            upsample(512, 512, 4, dropout_prob=dropout_prob),
            upsample(1024, 512, 4, dropout_prob=dropout_prob),
            upsample(1024, 512, 4, dropout_prob=dropout_prob),
            upsample(1024, 512, 4, dropout_prob=dropout_prob),
            upsample(1024, 256, 4, dropout_prob=dropout_prob),
            upsample(512, 128, 4, dropout_prob=dropout_prob),
            upsample(256, 64, 4, dropout_prob=dropout_prob)
        ])
        
        # original final layer
        self.last = nn.ConvTranspose2d(128, output_channels, 4, stride=2, padding=1)
        
        # Enhanced attention for encoder (downsampling path)
        encoder_channels = [64, 128, 256, 512, 512, 512, 512, 512]
        self.encoder_attentions = nn.ModuleList()
        
        for i, ch in enumerate(encoder_channels):
            if i >= 2:  
                if use_hierarchical_attention:
                    attention = HierarchicalAttention(ch, num_heads=min(num_heads, ch//32))
                else:
                    attention = FullSelfAttention(ch, num_heads=min(num_heads, ch//32))
                self.encoder_attentions.append(attention)
            else:
                self.encoder_attentions.append(None)
        
        # Enhanced attention for decoder (skip connections)
        decoder_channels = [512, 512, 512, 512, 256, 128, 64]
        encoder_skip_channels = [512, 512, 512, 512, 256, 128, 64]
        
        self.decoder_attentions = nn.ModuleList()
        self.cross_scale_attentions = nn.ModuleList()
        
        for i, (dec_ch, enc_ch) in enumerate(zip(decoder_channels, encoder_skip_channels)):
            # Skip connection attention
            if use_hierarchical_attention:
                skip_attention = HierarchicalAttention(enc_ch, num_heads=min(num_heads, enc_ch//32))
            else:
                skip_attention = FullSelfAttention(enc_ch, num_heads=min(num_heads, enc_ch//32))
            self.decoder_attentions.append(skip_attention)
            
            # Cross-scale attention between decoder and encoder
            if use_cross_scale_attention:
                cross_attn = CrossScaleAttention(enc_ch, dec_ch, num_heads=min(num_heads//2, enc_ch//32))
                self.cross_scale_attentions.append(cross_attn)
            else:
                self.cross_scale_attentions.append(None)
        
        # Residual blocks in bottleneck (after last downsample)
        if use_residual_blocks:
            self.bottleneck_blocks = nn.ModuleList([
                AttentionResidualBlock(512, num_heads=min(num_heads, 512//32))
                for _ in range(3)  
            ])
        
    def forward(self, x):
        # Encoder path (original downsampling with added attention)
        skips = []
        current = x
        
        for i, (conv_layer, attention) in enumerate(zip(self.conv_layers, self.encoder_attentions)):
            current = conv_layer(current)
            
            # attention in encoder
            if attention is not None:
                current = attention(current)
            
            skips.append(current)
        
        # Bottleneck processing
        if self.use_residual_blocks:
            for block in self.bottleneck_blocks:
                current = block(current)
        
        # Exclude bottleneck from skips
        skips = skips[:-1]
        
        # Decoder path (original upsampling with enhanced attention)
        for i, (up_layer, skip_attn, cross_attn) in enumerate(zip(
            self.up_layers, self.decoder_attentions, self.cross_scale_attentions)):
            
            # Upsample decoder features
            current = up_layer(current)
            
            # Get corresponding skip connection
            skip = skips[-(i + 1)]
            
            # Apply cross-scale attention (decoder queries skip)
            if cross_attn is not None:
                skip = cross_attn(skip, current)
            
            # Apply self-attention to skip connection
            skip = skip_attn(skip)
            
            # Concatenate (original approach)
            current = torch.cat([current, skip], dim=1)
        
        # Final output (original)
        output = self.last(current)
        return output






def load_model_state(file_path, **kwargs):


    model_state = PowerfulGenerator(**kwargs)  # H100 beast mode
    model_state.load_state_dict(torch.load(file_path, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state.eval()
    model_state.to(device)
    return model_state
