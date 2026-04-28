import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D convolutions.
    Dynamically weighs the importance of each channel (lead) based on global context.
    """

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class BasicBlock1d(nn.Module):
    """
    Standard ResNet Basic Block for 1D convolutions.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1d, self).__init__()

        # First convolutional layer of the block
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        # Second convolutional layer
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection (if dimensions change, we need a 1x1 conv to match them)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

        # Squeeze-and-Excitation block
        self.se = SEBlock(out_channels)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply Squeeze-and-Excitation
        out = self.se(out)

        # Residual connection! Add the original input back before the final ReLU
        out += identity
        out = F.relu(out)
        return out


class WFDBResNet(nn.Module):
    """
    A 1D ResNet Architecture for real PhysioNet WFDB EKG Classification.
    """

    def __init__(self, num_classes: int = 55):
        super(WFDBResNet, self).__init__()

        # 1. Initial Stem (Process the raw 12-lead signal)
        # 500Hz data means high frequency. We use a large kernel and stride to downsample initially.
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=12,
                out_channels=32,
                kernel_size=15,
                stride=3,
                padding=7,
                bias=False,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # 2. Residual Blocks
        self.layer1 = self._make_layer(
            in_channels=32, out_channels=64, num_blocks=2, stride=1
        )
        self.layer2 = self._make_layer(
            in_channels=64, out_channels=128, num_blocks=2, stride=2
        )
        self.layer3 = self._make_layer(
            in_channels=128, out_channels=256, num_blocks=2, stride=2
        )

        # 3. Global Average Pooling (Makes the network invariant to exact sequence length)
        # No matter how long the EKG is, it squashes the time dimension to 1.
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # 4. Classification Head
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        # The first block might have a stride > 1 to downsample
        layers.append(BasicBlock1d(in_channels, out_channels, stride))
        # The remaining blocks maintain the channel size
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1d(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)

        # Flatten: (batch_size, channels, 1) -> (batch_size, channels)
        x = x.view(x.size(0), -1)

        x = self.dropout(x)
        logits = self.fc(x)

        return logits


class WFDBResNetLSTM(nn.Module):
    """
    A CRNN (CNN + RNN) Architecture for real PhysioNet WFDB EKG Classification.
    Uses ResNet for feature extraction and an LSTM for sequence modeling.
    """

    def __init__(self, num_classes: int = 55):
        super(WFDBResNetLSTM, self).__init__()

        # 1. Initial Stem
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=12,
                out_channels=32,
                kernel_size=15,
                stride=3,
                padding=7,
                bias=False,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # 2. Residual Blocks
        self.layer1 = self._make_layer(
            in_channels=32, out_channels=64, num_blocks=2, stride=1
        )
        self.layer2 = self._make_layer(
            in_channels=64, out_channels=128, num_blocks=2, stride=2
        )
        self.layer3 = self._make_layer(
            in_channels=128, out_channels=256, num_blocks=2, stride=2
        )

        # 3. LSTM Layer
        # input_size=256 because that's the number of channels coming out of layer3
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # 4. Self-Attention Layer
        # 128 hidden_size * 2 directions = 256
        self.attention = nn.Linear(256, 1)

        # 5. Classification Head
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock1d(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1d(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 12, seq_len)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # ResNet output shape: (batch, channels=256, compressed_seq_len)
        # LSTM expects shape: (batch, seq_len, input_size) if batch_first=True
        x = x.permute(0, 2, 1)

        # Pass sequence through LSTM
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch, seq_len, 256)

        # Apply Self-Attention
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(
            attn_weights, dim=1
        )  # Normalize over sequence length

        # Multiply LSTM output by attention weights and sum over sequence
        # Context vector shape: (batch, 256)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)

        x = self.dropout(context_vector)
        logits = self.fc(x)

        return logits


if __name__ == "__main__":
    # Test the math with a dummy tensor
    # batch_size=4, 12 leads, 5000 samples (10 seconds at 500Hz)
    dummy_input = torch.randn(4, 12, 5000)
    model = WFDBResNet(num_classes=55)

    output = model(dummy_input)
    print(f"Model Architecture:\n{model}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Test variable length! Global Average Pooling should handle this perfectly!
    dummy_long_input = torch.randn(4, 12, 10000)  # 20 seconds!
    output_long = model(dummy_long_input)
    print(f"\nLong Input shape: {dummy_long_input.shape}")
    print(f"Long Output shape: {output_long.shape} (It still works!)")

    print("\n--- Testing WFDBResNetLSTM ---")
    model_lstm = WFDBResNetLSTM(num_classes=55)
    output_lstm = model_lstm(dummy_long_input)
    print(f"LSTM Output shape: {output_lstm.shape}")
