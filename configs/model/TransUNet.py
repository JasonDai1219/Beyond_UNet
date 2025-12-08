import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self, n_channels):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = self.pool(x1)
        x2 = F.relu(self.conv2(x2))
        x3 = self.pool(x2)
        x3 = F.relu(self.conv3(x3))
        cnn_feature = self.pool(x3)
        return x1, x2, x3, cnn_feature

class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransUNet(nn.Module):
    def __init__(self, n_channels, n_classes, nhead, num_transformer_layers):
        super(TransUNet, self).__init__()
        # Encode input image
        self.cnn_encoder = CNNEncoder(n_channels)
        # Transformer layers
        self.transformer = nn.Sequential(*[TransformerLayer(256, nhead) for _ in range(num_transformer_layers)])
        
        # Adjust channel size from D to 512
        self.conv_after_transformer = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # Upsample
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(512, 256, kernel_size=3, padding=1)  # Skip connection included
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, image):
        # Process image data
        x1, x2, x3, cnn_feature = self.cnn_encoder(image)
        # Flatten spatial dimensions
        cnn_feature_flat = cnn_feature.contiguous().view(cnn_feature.size(0), -1, cnn_feature.size(1)).permute(1, 0, 2)
        
        # Apply Transformer layers
        transformer_feature = self.transformer(cnn_feature_flat)
        transformer_feature = transformer_feature.permute(1, 0, 2)
        
        # Reshape back to image dimensions
        transformer_feature = transformer_feature.contiguous().view(cnn_feature.size(0), cnn_feature.size(1), 
                                                                    cnn_feature.size(2), cnn_feature.size(3))
        # Adjust channel size after transformer
        transformer_feature = self.conv_after_transformer(transformer_feature)

        # Decode with skip connections
        x = self.upconv1(transformer_feature)
        x = torch.cat((x, x3), dim=1)
        x = F.relu(self.conv4(x))
        
        x = self.upconv2(x)
        x = torch.cat((x, x2), dim=1)
        x = F.relu(self.conv5(x))

        x = self.upconv3(x)
        x = torch.cat((x, x1), dim=1)
        x = F.relu(self.conv6(x))

        x = self.final_conv(x)
        return x
