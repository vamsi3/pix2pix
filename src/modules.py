import torch
import torch.nn as nn

class EncoderDecoderBlock(nn.Module):
    def __init__(self, module_type, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, dropout_p=0.0, norm=True, activation=True):
        super().__init__()
        if module_type == 'encoder':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        elif module_type == 'decoder':
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        else:
            raise NotImplementedError(f"Module type '{module_type}' is not valid")

        self.lrelu = nn.LeakyReLU(0.2) if activation else None
        self.norm = nn.BatchNorm2d(out_channels) if norm else None
        self.dropout = nn.Dropout2d(dropout_p)

    def forward(self, x):
        x = self.lrelu(x) if self.lrelu else x
        x = self.conv(x)
        x = self.norm(x) if self.norm else x
        x = self.dropout(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, bias=False, dropout_p=0.5, norm=True):
        super().__init__()
        self.encoders = nn.ModuleList([
            EncoderDecoderBlock('encoder', in_channels, 64, bias=bias, norm=False, activation=False),
            EncoderDecoderBlock('encoder', 64, 128, bias=bias, norm=norm),
            EncoderDecoderBlock('encoder', 128, 256, bias=bias, norm=norm),
            EncoderDecoderBlock('encoder', 256, 512, bias=bias, norm=norm),
            EncoderDecoderBlock('encoder', 512, 512, bias=bias, norm=norm),
            EncoderDecoderBlock('encoder', 512, 512, bias=bias, norm=norm),
            EncoderDecoderBlock('encoder', 512, 512, bias=bias, norm=False)
        ])
        self.decoders = nn.ModuleList([
            EncoderDecoderBlock('decoder', 512, 512, bias=bias, norm=norm),
            EncoderDecoderBlock('decoder', 1024, 512, bias=bias, norm=norm),
            EncoderDecoderBlock('decoder', 1024, 512, bias=bias, norm=norm),
            EncoderDecoderBlock('decoder', 1024, 256, bias=bias, norm=norm),
            EncoderDecoderBlock('decoder', 512, 128, bias=bias, norm=norm),
            EncoderDecoderBlock('decoder', 256, 64, bias=bias, norm=norm),
            EncoderDecoderBlock('decoder', 128, out_channels, bias=bias, norm=False)
        ])

    def forward(self, x):
        encoder_outputs = [x]
        for encoder in self.encoders:
            encoder_outputs.append(encoder(encoder_outputs[-1]))
        for i, p in enumerate(zip(self.decoders, reversed(encoder_outputs))):
            decoder, encoder_out = p
            output = decoder(torch.cat([output, encoder_out], 1) if i > 0 else encoder_out)
        return nn.Tanh()(output)

class PatchGAN(nn.Module):
    def __init__(self, in_channels=6, out_channels=1, bias=False, norm=True):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.discriminator_blocks = nn.ModuleList([
            EncoderDecoderBlock('encoder', in_channels, 64, bias=bias, norm=False, activation=False),
            EncoderDecoderBlock('encoder', 64, 128, bias=bias, norm=norm),
            EncoderDecoderBlock('encoder', 128, 256, bias=bias, norm=norm),
            EncoderDecoderBlock('encoder', 256, 512, bias=bias, norm=norm, stride=1),
            EncoderDecoderBlock('encoder', 512, out_channels, bias=bias, norm=False, stride=1)
        ])

    def forward(self, x, cond):
        output = torch.cat([x, cond], 1)
        for block in self.discriminator_blocks:
            output = block(output)
        output = self.sigmoid(output)
        return output
