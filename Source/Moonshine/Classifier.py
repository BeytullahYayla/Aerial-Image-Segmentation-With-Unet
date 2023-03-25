import torch
from moonshine.models import UNet
class Classifier(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        # Create a blank model based on the available architectures.
        self.backbone = UNet(name="unet50_fmow_rgb")

        # If we are using pretrained weights, load them here. In
        # general, using the decoder weights isn't preferred unless
        # your downstream task is also a reconstruction task. We suggest
        # trying only the encoder first.
        if pretrained:
            self.backbone.load_weights(
                encoder_weights="unet50_fmow_rgb", decoder_weights=None
            )

        # Run a per-pixel classifier on top of the output vectors.
        self.classifier = torch.nn.Conv2d(32, 2, (1, 1))

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)



    




