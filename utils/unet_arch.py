
import torch
import torch.nn as nn
# U-Net architecture
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

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int): # F_g: number of input channels for the gate, F_l: number of channels for the encoder features (skip connection), F_int: number of intermediate channels
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi #attention-weighted skip connection

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Define the downsample layers
        self.conv_layers = nn.ModuleList([
            downsample(1, 64, 4),
            downsample(64, 128, 4),
            downsample(128, 256, 4),
            downsample(256, 512, 4),
            downsample(512, 512, 4),
            downsample(512, 512, 4),
            downsample(512, 512, 4),
            downsample(512, 512, 4)
        ])

        # Define the upsample layers
        self.up_layers = nn.ModuleList([
            upsample(512, 512, 4),
            upsample(1024, 512, 4),
            upsample(1024, 512, 4),
            upsample(1024, 512, 4),
            upsample(1024, 256, 4),
            upsample(512, 128, 4),
            upsample(256, 64, 4)
        ])

        # Final convolutional layer for generating the output
        self.last = nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1)
        
        # add attention blocks
        self.attn_blocks = nn.ModuleList([
            AttentionBlock(512, 512, 256),
            AttentionBlock(512, 512, 256),
            AttentionBlock(512, 512, 256),
            AttentionBlock(512, 512, 256),
            AttentionBlock(256, 256, 128),
            AttentionBlock(128, 128, 64),
            AttentionBlock(64, 64, 32)
        ])
        

    def forward(self, x):
        # Downsampling through the model
        skips = []
        for layer in self.conv_layers:
            x = layer(x)
            skips.append(x)

        
        skips = skips[:-1] # exclude bottleneck

        # Upsampling and establishing skip connections
        for i, (up, attn) in enumerate(zip(self.up_layers, self.attn_blocks)):
            x = up(x)
            skip = skips[-(i + 1)]
            gated_skip = attn(g=x, x=skip)
            x = torch.cat([x, gated_skip], dim=1)
            
        x = self.last(x)
        return x

def load_model_state(file_path):
    model_state = Generator()  # Assuming Generator is a defined class
    model_state.load_state_dict(torch.load(file_path, map_location=
    "cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state.eval()
    model_state.to(device)
    return model_state
