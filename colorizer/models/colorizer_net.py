import torch
from torch import nn
import torch.nn.functional as F


class LowLevelFeaturesNetwork(nn.Module):
    """Network that extracts low-level features from the image.

    Extracts basic features like edges, textures, and gradients through
    a series of convolutional layers with stride-2 downsampling.

    Input shape: (B, 1, 224, 224) - Grayscale L channel
    Output shape: (B, 512, 28, 28) - Low-level feature maps
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        return x


class GlobalFeaturesNetwork(nn.Module):
    """Network that extracts global semantic features.

    Captures high-level semantic information about the scene through
    additional downsampling and fully-connected layers.

    Input shape: (B, 512, 28, 28) - Low-level features
    Output shape: (B, 256) - Global feature vector for fusion
                 (B, 512) - Classification features (intermediate output)
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(7 * 7 * 512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(-1, 7 * 7 * 512)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        classification_input = x

        x = F.relu(self.fc3(x))

        return x, classification_input


class ClassNet(nn.Module):
    """Auxiliary classification network for scene categories.

    Performs scene classification as an auxiliary task to help the model
    learn better semantic representations for colorization.

    Input shape: (B, 512) - Classification features from GlobalFeaturesNetwork
    Output shape: (B, num_classes) - Class logits
    """

    def __init__(self, num_classes):
        super(ClassNet, self).__init__()

        self.num_classes = num_classes

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out


class MidLevelFeaturesNetwork(nn.Module):
    """Network that extracts mid-level features from the image.

    Processes low-level features to capture regional structure and patterns
    while maintaining spatial resolution for precise colorization.

    Input shape: (B, 512, 28, 28) - Low-level features
    Output shape: (B, 256, 28, 28) - Mid-level feature maps
    """

    def __init__(self):
        super().__init__()

        # Keep spatial resolution by padding
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class ColorizationNetwork(nn.Module):
    """Network that generates the final color prediction.

    Upsamples the fused features through a series of convolutional layers
    to produce the ab color channels at the original resolution.

    Input shape: (B, 256, 28, 28) - Fused mid-level and global features
    Output shape: (B, 2, 224, 224) - Predicted ab channels in [0, 1]
    """

    def __init__(self):
        super().__init__()

        # fusion layer output has 256 channels
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))

        # Upsample
        out = nn.functional.interpolate(input=out, scale_factor=2)
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))

        # Upsample
        out = nn.functional.interpolate(input=out, scale_factor=2)
        out = F.relu(self.conv4(out))
        # '''The output layer of the colorization network consists
        # of a convolutional layer with a Sigmoid transfer
        # function that outputs the chrominance of the
        # input grayscale image. '''
        out = torch.sigmoid(self.conv5(out))

        # Upsample
        out = nn.functional.interpolate(input=out, scale_factor=2)

        return out


class LTBCNetwork(nn.Module):
    """Complete "Let There Be Color" network for automatic image colorization.

    Combines low-level, mid-level, and global features through a fusion layer
    to predict color channels. Includes an auxiliary classification task for
    improved semantic understanding.

    Input shape: (B, 1, 224, 224) - Grayscale L channel in [0, 1]
    Output shape:
        - (B, 2, 224, 224) - Predicted ab channels in [0, 1]
        - (B, num_classes) - Scene classification logits
    """

    def __init__(self, num_classes: int = 365):
        """Initialize the network.

        Args:
            num_classes: Number of scene categories for classification (default: 365 for Places365)
        """
        super().__init__()

        self.conv_fuse = nn.Conv2d(512, 256, kernel_size=1)

        self.low = LowLevelFeaturesNetwork()
        self.mid = MidLevelFeaturesNetwork()
        self.classifier = ClassNet(num_classes)
        self.glob = GlobalFeaturesNetwork()
        self.col = ColorizationNetwork()

    def fusion_layer(self, mid_out, glob_out):
        """Fuse mid-level spatial features with global semantic features.

        Replicates the global feature vector across spatial dimensions and
        concatenates with mid-level features, then applies 1x1 convolution.

        Args:
            mid_out: (B, 256, H, W) - Mid-level features
            glob_out: (B, 256) - Global features

        Returns:
            (B, 256, H, W) - Fused features
        """
        h = mid_out.shape[2]
        w = mid_out.shape[3]

        # Create 3D volume from global features
        glob_stack2d = torch.stack(tuple(glob_out for _ in range(w)), 1)
        glob_stack3d = torch.stack(tuple(glob_stack2d for _ in range(h)), 1)
        glob_stack3d = glob_stack3d.permute(0, 3, 1, 2)

        # 'Merge' two volumes
        stack_volume = torch.cat((mid_out, glob_stack3d), 1)

        out = F.relu(self.conv_fuse(stack_volume))
        return out

    def forward(self, x):
        """Forward pass through the complete colorization network.

        Args:
            x: (B, 1, 224, 224) - Input grayscale L channel

        Returns:
            Tuple of:
                - (B, 2, 224, 224) - Predicted ab channels
                - (B, num_classes) - Classification logits
        """
        # Low level
        low_out = self.low(x)

        # Net branch
        mid_out = low_out
        glob_out = low_out

        # Mid level
        mid_out = self.mid(mid_out)

        # Global
        glob_out, classification_input = self.glob(glob_out)

        classification_output = self.classifier(classification_input)

        # Fusion layer
        out = self.fusion_layer(mid_out, glob_out)

        # Colorization Net
        out = self.col(out)

        return out, classification_output
