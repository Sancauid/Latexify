import torch
import torch.nn as nn


class LatexifyModel(nn.Module):
    def __init__(self, encoder, decoder, name) -> None:
        super(LatexifyModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.name = name

        # report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        n_params_enc = sum(p.numel() for p in self.encoder.parameters())
        n_params_dec = sum(p.numel() for p in self.decoder.parameters())
        print("Number of parameters in total: %.2fM" % (n_params / 1e6,))
        print("  - Encoder: %.2fM" % (n_params_enc / 1e6,))
        print("  - Decoder: %.2fM" % (n_params_dec / 1e6,))

    def forward(self, idx, images, targets=None):
        """Take in and process masked src and target sequences.

        Args:
            images: sequence to the decoder ([B, C, H, W])
            idx: sequence to the decoder ([B, Th, D])
            targets: target output ([B, Ti])

        Note:

        1. The idx should contain 1 <START> token for encdec architecture.
        It can be empty for gpt architecture.
        """
        encoded_img = self.encoder(images)
        encoded_img = torch.flatten(encoded_img, 2)
        encoded_img = torch.transpose(encoded_img, 1, 2)
        return self.decoder(idx, encoded_img, targets)
