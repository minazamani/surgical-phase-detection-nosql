import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedResidualLayer(nn.Module):

    def __init__(
        self,
        dilation,
        in_channels,
        out_channels,
        dropout=0.5,
    ):
        super().__init__()

        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )

        self.conv_1x1 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=1,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        out = self.conv_dilated(x)
        out = F.relu(out)

        out = self.conv_1x1(out)
        out = self.dropout(out)

        return (x + out) * mask


class SingleStageModel(nn.Module):

    def __init__(
        self,
        num_layers,
        num_f_maps,
        input_dim,
        num_classes,
    ):
        super().__init__()

        self.conv_in = nn.Conv1d(
            input_dim,
            num_f_maps,
            kernel_size=1,
        )

        self.layers = nn.ModuleList(
            [
                DilatedResidualLayer(
                    dilation=2 ** i,
                    in_channels=num_f_maps,
                    out_channels=num_f_maps,
                )
                for i in range(num_layers)
            ]
        )

        self.conv_out = nn.Conv1d(
            num_f_maps,
            num_classes,
            kernel_size=1,
        )

    def forward(self, x, mask):

        out = self.conv_in(x)

        for layer in self.layers:
            out = layer(out, mask)

        out = self.conv_out(out)

        return out * mask


class MultiStageModel(nn.Module):

    def __init__(
        self,
        num_stages,
        num_layers,
        num_f_maps,
        input_dim,
        num_classes,
    ):
        super().__init__()

        self.stage1 = SingleStageModel(
            num_layers=num_layers,
            num_f_maps=num_f_maps,
            input_dim=input_dim,
            num_classes=num_classes,
        )

        self.stages = nn.ModuleList(
            [
                SingleStageModel(
                    num_layers=num_layers,
                    num_f_maps=num_f_maps,
                    input_dim=num_classes,
                    num_classes=num_classes,
                )
                for _ in range(num_stages - 1)
            ]
        )

    def forward(self, x, mask):

        out = self.stage1(
            x,
            mask
        )

        outputs = [
            out
        ]

        for stage in self.stages:

            out = F.softmax(
                out,
                dim=1
            )

            out = stage(
                out,
                mask
            )

            outputs.append(
                out
            )

        return torch.stack(
            outputs,
            dim=0
        )