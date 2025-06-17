
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

    def forward(self, x):
        # Downsampling through the model
        skips = []
        for layer in self.conv_layers:
            x = layer(x)
            skips.append(x)

        skips = skips[:-1]

        # Upsampling and establishing skip connections
        for layer, skip in zip(self.up_layers, reversed(skips)):
            x = layer(x)
            x = torch.cat([x, skip], dim=1)

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
