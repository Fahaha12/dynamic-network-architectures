import torch
from torch import nn
import numpy as np
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op, get_matching_convtransp
from dynamic_network_architectures.building_blocks.all_attention import *


class PlainConvEncoder(nn.Module):
    has_shown_prompt = [False]  # 类属性，用于记录是否已经显示过提示
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 nonlin_first: bool = False,
                 pool: str = 'conv',
                 g: int = 8  # g for DualConv
                 ):

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                             "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        stages = []
        for s in range(n_stages):
            stage_modules = []
            if pool == 'max' or pool == 'avg':
                if (isinstance(strides[s], int) and strides[s] != 1) or \
                        isinstance(strides[s], (tuple, list)) and any([i != 1 for i in strides[s]]):
                    stage_modules.append(get_matching_pool_op(conv_op, pool_type=pool)(kernel_size=strides[s], stride=strides[s]))
                conv_stride = 1
            elif pool == 'conv':
                conv_stride = strides[s]
            else:
                raise RuntimeError()
            stage_modules.append(nn.Sequential(*[
                DualConvBlock(
                    input_channels if i == 0 else features_per_stage[s],
                    features_per_stage[s],
                    stride=conv_stride if i == 0 else 1,
                    g=g,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs
                )
                for i in range(n_conv_per_stage[s])
            ]))

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

        # 为所有阶段添加MBTFE模块
        self.mbtfe_modules = nn.ModuleList([
            MBTFE(features_per_stage[i], features_per_stage[i])
            for i in range(n_stages)
        ])

        ################################################## 边缘聚合块 ##################################################
        transpconv_op = get_matching_convtransp(conv_op=self.conv_op)
        self.downblock_channal = [32, 64, 128, 256, 512, 512]
        self.mattn = Spartial_Attention3d(kernel_size=3)
        self.mdcat1 = nn.Sequential(
            StackedDualConvBlocks(
                1, self.downblock_channal[0], self.downblock_channal[0], kernel_sizes[s], 2, g,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first))

        self.mdcat2 = nn.Sequential(
            StackedDualConvBlocks(
                1, self.downblock_channal[0] + self.downblock_channal[1], 
                self.downblock_channal[0] + self.downblock_channal[1], kernel_sizes[s], 2, g,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first))

        self.mupcat3 = nn.Sequential(
            StackedDualConvBlocks(
                1, self.downblock_channal[0] + self.downblock_channal[1] + self.downblock_channal[2], 
                self.downblock_channal[2], kernel_sizes[s], 1, g,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first))
        self.gate3 = Gate(self.downblock_channal[2], self.downblock_channal[2])
        self.mupcat2 = transpconv_op(self.downblock_channal[0] + self.downblock_channal[1] + self.downblock_channal[2],
                                     self.downblock_channal[1], kernel_size=2, stride=2, bias=False)
        self.gate2 = Gate(in_channels=self.downblock_channal[1], out_channels=self.downblock_channal[1])
        self.mupcat1 = transpconv_op(self.downblock_channal[0] + self.downblock_channal[1] + self.downblock_channal[2],
                                     self.downblock_channal[0], kernel_size=4, stride=4, bias=False)
        self.gate1 = Gate(in_channels=self.downblock_channal[0], out_channels=self.downblock_channal[0])
        ################################################## 边缘聚合块 ##################################################

    def forward(self, x):
        ret = []
        for i, s in enumerate(self.stages):
            x = s(x)
            x = self.mbtfe_modules[i](x)
            ret.append(x)

        if not PlainConvEncoder.has_shown_prompt[0]:  # 如果还未显示过提示
            print("################################################## EAB&MBTFE ##################################################")
            PlainConvEncoder.has_shown_prompt[0] = True  # 将提示标记为已显示
        # middle attention
        m1 = self.mattn(ret[0])
        m2 = self.mattn(ret[1])
        m3 = self.mattn(ret[2])

        m1m2 = torch.cat([self.mdcat1(m1), m2], dim=1)  # Shape : [B, C=32+64, D/2, H/2, W/2]
        m_feature = torch.cat([self.mdcat2(m1m2), m3], dim=1)  # Shape : [B, C=32+64+128, D/4, H/4, W/4]

        ret[0] = self.gate1(self.mupcat1(m_feature), ret[0])
        ret[1] = self.gate2(self.mupcat2(m_feature), ret[1])
        ret[2] = self.gate3(self.mupcat3(m_feature), ret[2])

        '''
        tensors = {'m1': m1, 'm2': m2, 'm3': m3, 'm1m2': m1m2, 'm_feature': m_feature, 'gate_output1': gate_output1, 'gate_output2': gate_output2, 'self.mupcat3(m_feature)': self.mupcat3(m_feature)}
        for name, tensor in tensors.items():
            print(f"Name: {name}, Shape: {tensor.shape}")
        '''

        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        output = np.int64(0)
        for s in range(len(self.stages)):
            if isinstance(self.stages[s], nn.Sequential):
                for sq in self.stages[s]:
                    if hasattr(sq, 'compute_conv_feature_map_size'):
                        output += self.stages[s][-1].compute_conv_feature_map_size(input_size)
            else:
                output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output

class MBTFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MBTFE, self).__init__()

        # 多尺度特征提取
        # 1×1×1层
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.prelu1 = nn.PReLU(out_channels)

        # 3×3×3层
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.prelu3 = nn.PReLU(out_channels)

        # 5×5×5层
        self.conv5_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm3d(out_channels)
        self.prelu5_1 = nn.PReLU(out_channels)

        self.conv5_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm3d(out_channels)
        self.prelu5_2 = nn.PReLU(out_channels)

        # 7×7×7层
        self.conv7_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn7_1 = nn.BatchNorm3d(out_channels)
        self.prelu7_1 = nn.PReLU(out_channels)

        self.conv7_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn7_2 = nn.BatchNorm3d(out_channels)
        self.prelu7_2 = nn.PReLU(out_channels)

        self.conv7_3 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn7_3 = nn.BatchNorm3d(out_channels)
        self.prelu7_3 = nn.PReLU(out_channels)

        # 相关性计算
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(out_channels, out_channels // 2)
        self.fc2 = nn.Linear(out_channels // 2, 4)  # 输出4个权重，对应4个分支

        # 特征选取
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 多尺度特征提取
        out1 = self.prelu1(self.bn1(self.conv1(x)))
        out2 = self.prelu3(self.bn3(self.conv3(x)))
        out3 = self.prelu5_2(self.bn5_2(self.conv5_2(self.prelu5_1(self.bn5_1(self.conv5_1(x))))))
        out4 = self.prelu7_3(self.bn7_3(self.conv7_3(self.prelu7_2(self.bn7_2(self.conv7_2(self.prelu7_1(self.bn7_1(self.conv7_1(x)))))))))

        # 相关性计算
        pool = self.avg_pool(out1 + out2 + out3 + out4)
        flat = torch.flatten(pool, 1)
        weight = self.fc2(self.fc1(flat))
        weight_map = self.softmax(weight)

        # 特征选取
        out1 = weight_map[:, 0].view(-1, 1, 1, 1, 1) * out1
        out2 = weight_map[:, 1].view(-1, 1, 1, 1, 1) * out2
        out3 = weight_map[:, 2].view(-1, 1, 1, 1, 1) * out3
        out4 = weight_map[:, 3].view(-1, 1, 1, 1, 1) * out4
        out = out1 + out2 + out3 + out4

        return out

class DualConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, g):
        super(DualConv, self).__init__()
        self.gc = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=g, bias=False)
        self.pwc = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, input_data):
        return self.gc(input_data) + self.pwc(input_data)        

class DualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, g, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs):
        super(DualConvBlock, self).__init__()
        self.conv = DualConv(in_channels, out_channels, stride, g)
        self.norm = norm_op(out_channels, **norm_op_kwargs) if norm_op else None
        self.dropout = dropout_op(**dropout_op_kwargs) if dropout_op else None
        self.nonlin = nonlin(**nonlin_kwargs) if nonlin else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.dropout:
            x = self.dropout(x)
        if self.nonlin:
            x = self.nonlin(x)
        return x        

class StackedDualConvBlocks(nn.Module):
    def __init__(self,
                 num_convs: int,
                 input_channels: int,
                 output_channels: Union[int, List[int], Tuple[int, ...]],
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 initial_stride: Union[int, List[int], Tuple[int, ...]],
                 g: int,
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 nonlin_first: bool = False
                 ):
        super().__init__()
        if not isinstance(output_channels, (tuple, list)):
            output_channels = [output_channels] * num_convs

        self.convs = nn.Sequential(
            DualConvBlock(
                input_channels, output_channels[0], initial_stride, g, norm_op,
                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
            ),
            *[
                DualConvBlock(
                    output_channels[i - 1], output_channels[i], 1, g, norm_op,
                    norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs, nonlin_first
                )
                for i in range(1, num_convs)
            ]
        )

        self.output_channels = output_channels[-1]
        self.initial_stride = maybe_convert_scalar_to_list(nn.Conv3d, initial_stride)  # 假设使用3D卷积

    def forward(self, x):
        return self.convs(x)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == len(self.initial_stride), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        output = self.convs[0].conv.compute_conv_feature_map_size(input_size)
        size_after_stride = [i // j for i, j in zip(input_size, self.initial_stride)]
        for b in self.convs[1:]:
            output += b.conv.compute_conv_feature_map_size(size_after_stride)
        return output 