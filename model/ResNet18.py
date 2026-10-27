import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# Conv + BN + ReLU
class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, **kwargs)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# ResNet block with skip connection
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNRelu(in_channels, out_channels, 3, stride=stride, padding=padding),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=padding),
            nn.BatchNorm3d(out_channels),
        )

        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.downsample = Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.downsample(x)
        x = self.block(x)
        x = self.relu(x + res)
        return x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class ResNet18(nn.Module):
    def __init__(self, in_channels=1, num_class=2):
        super(ResNet18, self).__init__()
        self.num_class = num_class

        self.conv1 = nn.Sequential(
            ConvBNRelu(in_channels, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, 1, 1)
        self.layer2 = self._make_layer(64, 128, 1, 2)
        self.layer3 = self._make_layer(128, 256, 1, 2)
        self.layer4 = self._make_layer(256, 512, 1, 2)

        # self.avgpool = nn.AdaptiveAvgPool3d(1)

    def _make_layer(self, in_channels, out_channels, n_blocks, stride=1):
        layer_list = []
        layer_list.append(Block(in_channels, out_channels, stride))
        for i in range(1, n_blocks):
            layer_list.append(Block(out_channels, out_channels))
        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # N, C, T, H, W
        # x = self.avgpool(x) # N, C
        return x


class CP_ResNet18(nn.Module):
    def __init__(self, in_channels=1, num_class=2):
        super(CP_ResNet18, self).__init__()
        self.in_channel = in_channels
        self.num_class = num_class
        self.C_feature = ResNet18(in_channels=self.in_channel, num_class=self.num_class)
        self.P_feature = ResNet18(in_channels=self.in_channel, num_class=self.num_class)

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, self.num_class),
            # nn.ReLU()
        )

        # self.fc3 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(256, self.num_class)
        # )

    def forward(self, x, y):
        x = self.C_feature(x)
        y = self.P_feature(y)
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        out = torch.cat((x, y), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        # out = self.fc3(out)
        return out

# ScaledDotProductAttention class
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(Q.size(-1))  # Q.size(-1) corresponds to d_k
        # Apply Softmax directly (no masking needed)
        attn = nn.Softmax(dim=-1)(scores)
        # Compute the context vector
        context = torch.matmul(attn, V)
        return context, attn


# MultiHeadAttention class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, n_heads, d_v):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(n_heads * d_v, d_model * 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(d_model * 4, d_model, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V):
        batch_size, N, C = input_Q.shape
        residual = input_Q
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # attn_mask is not used
        context, attn = ScaledDotProductAttention()(Q, K, V)
        # context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        output = self.fc(context)
        output = self.layer_norm(output + residual)

        # Return the output with residual connection and normalization
        # return nn.LayerNorm(output.size(-1)).cuda()(output + residual), attn
        return output, attn


class CP_ResNet18_mutli_attention(nn.Module):
    def __init__(self, in_channels=1, num_class=2, d_model=512, d_k=64, n_heads=8, d_v=64):
        super(CP_ResNet18_mutli_attention, self).__init__()
        self.in_channel = in_channels
        self.num_class = num_class

        # ResNet18 feature extractors for C and P
        self.C_feature = ResNet18(in_channels=self.in_channel, num_class=self.num_class)
        self.P_feature = ResNet18(in_channels=self.in_channel, num_class=self.num_class)

        # MultiHead Attention
        self.multihead_attn = MultiHeadAttention(d_model=d_model, d_k=d_k, n_heads=n_heads, d_v=d_v)  # MultiHeadAttention(512, 64, 8, 64)

        # self.fc1 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 512),
        #     nn.ReLU()
        # )

        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, self.num_class),
        )

    def forward(self, x, y):
        # Extract features from ResNet18
        x = self.C_feature(x)  # C feature (Query)
        y = self.P_feature(y)  # P feature (Key and Value)

        # Flatten the feature maps
        b, c, h, w, d = x.shape

        x = x.flatten(2).transpose(1, 2)
        y = y.flatten(2).transpose(1, 2)

        # Flatten the output of ResNet18
        # x = x.view(x.size(0), -1)  # Flatten x for attention (Query)
        # y = y.view(y.size(0), -1)  # Flatten y for attention (Key and Value)

        # Apply MultiHead Attention on x (Q) and y (K, V)
        attn_output, attn_weights = self.multihead_attn(y, x, x)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(b, c, h, w, d)
        attn_output = attn_output.mean([2, 3, 4])  # Global average pooling
        # Feed the attention output to the fully connected layers
        out = self.fc2(attn_output)
        # out = self.fc2(out)

        return out

# Single-modality attention model
class CT_ResNet_SelfAttention(nn.Module):
    def __init__(self, in_channels=1, num_class=1, d_model=512, n_heads=8):
        super(CT_ResNet_SelfAttention, self).__init__()

        self.in_channel = in_channels
        self.num_class = num_class

        # Backbone network
        self.resnet = ResNet18(in_channels=self.in_channel, num_class=self.num_class)

        # Adaptive dimension adjustment
        self.channel_adjust = nn.Conv3d(512, d_model, kernel_size=1)  # Align channel count to d_model

        # Multi-head self-attention
        self.multihead_attn = MultiHeadAttention(d_model=d_model, d_k=d_model // n_heads,
                                                 n_heads=n_heads, d_v=d_model // n_heads)

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(d_model, num_class)
        )

    def forward(self, x):
        # Feature extraction
        x = self.resnet(x)  # Output shape: (b, 512, D, H, W)
        x = self.channel_adjust(x)  # Adjust channels -> (b, d_model, D, H, W)

        # Prepare attention input
        b, c, d, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # Shape: (b, d*h*w, d_model)

        # Self-attention (Q=K=V)
        attn_out, _ = self.multihead_attn(x_flat, x_flat, x_flat)

        # Restore spatial dimensions
        attn_out = attn_out.transpose(1, 2).reshape(b, c, d, h, w)

        # Classification
        return self.classifier(attn_out)

class CP_ResNet10_early(nn.Module):
    def __init__(self, in_channels=2, num_class=2):
        super(CP_ResNet10_early, self).__init__()
        self.num_class = num_class
        self.conv1 = nn.Sequential(
            ConvBNRelu(in_channels, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3)),
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64, 64, 1, 1)
        self.layer2 = self._make_layer(64, 128, 1, 2)
        self.layer3 = self._make_layer(128, 256, 1, 2)
        self.layer4 = self._make_layer(256, 512, 1, 2)

        self.avgpool = nn.AdaptiveAvgPool3d(1)

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, self.num_class)
        )

    def _make_layer(self, in_channels, out_channels, n_blocks, stride=1):
        layer_list = []
        layer_list.append(Block(in_channels, out_channels, stride))
        for i in range(1, n_blocks):
            layer_list.append(Block(out_channels, out_channels))
        return nn.Sequential(*layer_list)

    def forward(self, x):
        # out = torch.cat([x, y], dim=1)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # N, C, T, H, W
        x = self.avgpool(x)  # N, C
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class Multi_resNet(nn.Module):
    def __init__(self, in_channels=1, num_class=2):
        super(Multi_resNet, self).__init__()
        self.in_channel = in_channels
        self.num_class = num_class
        self.C_feature = ResNet18(in_channels=self.in_channel, num_class=self.num_class)
        self.P_feature = ResNet18(in_channels=self.in_channel, num_class=self.num_class)

        self.T_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, self.num_class)
        )

        self.N_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, self.num_class)
        )

    def forward(self, x, y):
        x = self.C_feature(x)
        y = self.P_feature(y)
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        out = torch.cat((x, y), dim=1)
        out1 = self.T_fc(out)
        out2 = self.N_fc(out)
        return out1, out2


class CT_ResNet_SelfAttention_MultiLabel(nn.Module):
    def __init__(self, in_channels=1, num_class=1, d_model=512, n_heads=8):
        super(CT_ResNet_SelfAttention_MultiLabel, self).__init__()

        self.in_channel = in_channels
        self.num_class = num_class

        # Backbone network
        self.resnet = ResNet18(in_channels=self.in_channel, num_class=self.num_class)

        # Adaptive dimension adjustment
        self.channel_adjust = nn.Conv3d(512, d_model, kernel_size=1)  # Align channel count to d_model

        # Multi-head self-attention
        self.multihead_attn = MultiHeadAttention(d_model=d_model, d_k=d_model // n_heads,
                                                 n_heads=n_heads, d_v=d_model // n_heads)

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(d_model, num_class)
        )

        self.p = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(d_model, num_class)
        )
        self.linba = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(d_model, num_class)
        )
        self.maishen = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(d_model, num_class)
        )
        self.xiongmo = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(d_model, num_class)
        )
        self.qidao = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(d_model, num_class)
        )
        self.g = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(d_model, num_class)
        )

    def forward(self, x):
        # Feature extraction
        x = self.resnet(x)  # Output shape: (b, 512, D, H, W)
        x = self.channel_adjust(x)  # Adjust channels -> (b, d_model, D, H, W)

        # Prepare attention input
        b, c, d, h, w = x.shape
        x_flat = x.flatten(2).transpose(1, 2)  # Shape: (b, d*h*w, d_model)

        # Self-attention (Q=K=V)
        attn_out, _ = self.multihead_attn(x_flat, x_flat, x_flat)

        # Restore spatial dimensions
        attn_out = attn_out.transpose(1, 2).reshape(b, c, d, h, w)

        # out = self.classifier(attn_out)

        out = [
            self.p(attn_out),  # Main invasiveness task output
            self.linba(attn_out),  # linba task output
            self.maishen(attn_out),  # maishen task output
            self.xiongmo(attn_out),  # xiongmo task output
            self.qidao(attn_out),  # qidao task output
            self.g(attn_out)  # g task output
        ]

        # Classification
        return out

class CP_ResNet18_mutli_attention_MultiLabel(nn.Module):
    def __init__(self, in_channels=1, num_class=2, d_model=512, d_k=64, n_heads=8, d_v=64):
        super(CP_ResNet18_mutli_attention_MultiLabel, self).__init__()
        self.in_channel = in_channels
        self.num_class = num_class

        # ResNet18 feature extractors for C and P
        self.C_feature = ResNet18(in_channels=self.in_channel, num_class=self.num_class)
        self.P_feature = ResNet18(in_channels=self.in_channel, num_class=self.num_class)

        # MultiHead Attention
        self.multihead_attn = MultiHeadAttention(d_model=d_model, d_k=d_k, n_heads=n_heads, d_v=d_v)  # MultiHeadAttention(512, 64, 8, 64)

        # self.fc1 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 512),
        #     nn.ReLU()
        # )

        # self.fc2 = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(512, self.num_class),
        # )

        self.p = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, self.num_class),
        )
        self.linba = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, self.num_class),
        )
        self.maishen = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, self.num_class),
        )
        self.xiongmo = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, self.num_class),
        )
        self.qidao = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, self.num_class),
        )
        self.g = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, self.num_class),
        )

        # Define task output heads, one logit per task
        # self.task_heads = nn.ModuleDict({
        #     'linba': nn.Linear(d_model, 1),  # linba task
        #     'maiguan': nn.Linear(d_model, 1),  # maiguan task
        #     'shenjing': nn.Linear(d_model, 1),  # shenjing task
        #     'xiongmo': nn.Linear(d_model, 1),  # xiongmo task
        #     'qidao': nn.Linear(d_model, 1)  # qidao task
        # })

    def forward(self, x, y):
        # Extract features from ResNet18
        x = self.C_feature(x)  # C feature (Query)
        y = self.P_feature(y)  # P feature (Key and Value)

        # Flatten the feature maps
        b, c, h, w, d = x.shape

        x = x.flatten(2).transpose(1, 2)
        y = y.flatten(2).transpose(1, 2)

        # Flatten the output of ResNet18
        # x = x.view(x.size(0), -1)  # Flatten x for attention (Query)
        # y = y.view(y.size(0), -1)  # Flatten y for attention (Key and Value)

        # Apply MultiHead Attention on x (Q) and y (K, V)
        attn_output, attn_weights = self.multihead_attn(y, x, x)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(b, c, h, w, d)
        attn_output = attn_output.mean([2, 3, 4])  # Global average pooling
        # Feed the attention output to the fully connected layers
        # out = self.fc2(attn_output)
        # out = self.fc2(out)

        # out = {
        #     'linba': self.linba(attn_output),
        #     'maiguan': self.maiguan(attn_output),
        #     'shenjing': self.shenjing(attn_output),
        #     'xiongmo': self.xiongmo(attn_output),
        #     'qidao': self.qidao(attn_output)
        # }

        # Output as a list with fixed task order
        out = [
            self.p(attn_output),  # Main invasiveness task output
            self.linba(attn_output),  # linba task output
            self.maishen(attn_output),  # maishen task output
            self.xiongmo(attn_output),  # xiongmo task output
            self.qidao(attn_output),  # qidao task output
            self.g(attn_output)  # g task output
        ]

        # out = {task: head(attn_output) for task, head in self.task_heads.items()}

        return out

if __name__ == '__main__':
    # model = CP_ResNet10_early(2, num_class=2)
    # model = Multi_resNet()
    model = CP_ResNet18_mutli_attention(1, 1)
    # print(model)
    summary(model, [(1, 1, 128, 128, 128), (1, 1, 128, 128, 128)])
