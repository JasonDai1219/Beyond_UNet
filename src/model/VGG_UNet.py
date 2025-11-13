import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv (No Skip Connection)"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(out_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class VGG_Autoencoder(nn.Module):
    def __init__(self, n_classes, bilinear=True, freeze_encoder=True):
        super(VGG_Autoencoder, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Load pre-trained VGG-16 and use its features as the encoder
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        encoder_features = vgg16.features

        # Define encoder blocks based on VGG-16 architecture
        self.encoder1 = encoder_features[:5]  # Output channels: 64
        self.encoder2 = encoder_features[5:10] # Output channels: 128
        self.encoder3 = encoder_features[10:17]# Output channels: 256
        self.encoder4 = encoder_features[17:24]# Output channels: 512
        self.encoder5 = encoder_features[24:] # Output channels: 512 (Bottleneck)

        # Freeze encoder weights if requested
        if freeze_encoder:
            for block in [self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.encoder5]:
                for param in block.parameters():
                    param.requires_grad = False

        # Define decoder blocks without skip connections
        # Use 5 up-sampling stages to mirror the 5 pooling stages in VGG-16
        # Channel plan: 512 -> 512 -> 512 -> 256 -> 128 -> 64
        self.up1 = Up(512, 512, bilinear)
        self.up2 = Up(512, 512, bilinear)
        self.up3 = Up(512, 256, bilinear)
        self.up4 = Up(256, 128, bilinear)
        self.up5 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # save original spatial size
        orig_h, orig_w = x.shape[2], x.shape[3]

        # Encoder path
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        x = self.encoder5(x) # Bottleneck

        # Decoder path (no skip connections)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        logits = self.outc(x)

        # Resize logits to match original input size (H, W) using bilinear
        # interpolation to handle cases where input size is not divisible by 32
        logits = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
        return logits

if __name__ == '__main__':
    # A simple test to check the model's input and output shapes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_classes = 104 # Example number of classes
    # Use user's desired input size: (H=image_height, W=image_width) = (250, 450)
    input_tensor = torch.randn(2, 3, 250, 450).to(device) # Batch size 2, 3 channels, 250x450 image
    
    # Test with encoder frozen (default)
    print("Testing with frozen encoder...")
    model_frozen = VGG_Autoencoder(n_classes=num_classes, freeze_encoder=True).to(device)
    output_frozen = model_frozen(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_frozen.shape}")
    assert output_frozen.shape == (2, num_classes, 250, 450), "Output shape is incorrect!"
    
    # Check if encoder weights are actually frozen
    for name, param in model_frozen.named_parameters():
        if 'encoder' in name:
            assert not param.requires_grad, f"Parameter {name} is not frozen!"
    print("Encoder weights are correctly frozen.")
    print("Model test passed for frozen encoder!")

    # Test with encoder unfrozen
    print("\nTesting with trainable encoder...")
    model_trainable = VGG_Autoencoder(n_classes=num_classes, freeze_encoder=False).to(device)
    output_trainable = model_trainable(input_tensor)
    
    # Check if encoder weights are trainable
    for name, param in model_trainable.named_parameters():
        if 'encoder' in name:
            assert param.requires_grad, f"Parameter {name} is not trainable!"
    print("Encoder weights are correctly set to trainable.")
    print("Model test passed for trainable encoder!")
