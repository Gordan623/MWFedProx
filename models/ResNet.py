import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x):
        return x * self.weight


def batch_norm(num_channels, bn_bias_init=None, bn_bias_freeze=False,
               bn_weight_init=None, bn_weight_freeze=False):
    m = nn.BatchNorm2d(num_channels)
    if bn_bias_init is not None:
        m.bias.data.fill_(bn_bias_init)
    if bn_bias_freeze:
        m.bias.requires_grad = False
    if bn_weight_init is not None:
        m.weight.data.fill_(bn_weight_init)
    if bn_weight_freeze:
        m.weight.requires_grad = False

    return m


class ConvBN(nn.Module):
    def __init__(self, do_batchnorm, c_in, c_out, bn_weight_init=1.0, pool=None, **kw):
        super().__init__()
        self.pool = pool
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1,
                              padding=1, bias=False)
        if do_batchnorm:
            self.bn = batch_norm(c_out, bn_weight_init=bn_weight_init, **kw)
        self.do_batchnorm = do_batchnorm
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.conv(x)
        if self.do_batchnorm:
            out = self.bn(out)
        out = self.relu(out)
        if self.pool:
            out = self.pool(out)
        return out

    def prep_finetune(self, iid, c_in, c_out, bn_weight_init=1.0, pool=None, **kw):
        self.bn.bias.requires_grad = False
        self.bn.weight.requires_grad = False
        layers = [self.conv]
        for l in layers:
            for p in l.parameters():
                p.requires_grad = True
        return itertools.chain.from_iterable([l.parameters() for l in layers])


class Residual(nn.Module):
    def __init__(self, do_batchnorm, c, **kw):
        super().__init__()
        self.res1 = ConvBN(do_batchnorm, c, c, **kw)
        self.res2 = ConvBN(do_batchnorm, c, c, **kw)

    def forward(self, x):
        return x + F.relu(self.res2(self.res1(x)))

    def prep_finetune(self, iid, c, **kw):
        layers = [self.res1, self.res2]
        return itertools.chain.from_iterable([l.prep_finetune(iid, c, c, **kw) for l in layers])


class BasicNet(nn.Module):
    def __init__(self, do_batchnorm, channels, weight, pool, num_classes,
                 initial_channels=3, new_num_classes=None, **kw):
        super().__init__()
        self.new_num_classes = new_num_classes
        self.prep = ConvBN(do_batchnorm, initial_channels, channels['prep'], **kw)

        self.layer1 = ConvBN(do_batchnorm, channels['prep'], channels['layer1'], pool=pool, **kw)
        self.res1 = Residual(do_batchnorm, channels['layer1'], **kw)

        self.layer2 = ConvBN(do_batchnorm, channels['layer1'], channels['layer2'], pool=pool, **kw)

        self.layer3 = ConvBN(do_batchnorm, channels['layer2'], channels['layer3'], pool=pool, **kw)
        self.res3 = Residual(do_batchnorm, channels['layer3'], **kw)

        self.pool = nn.MaxPool2d(4)
        self.linear = nn.Linear(channels['layer3'], num_classes, bias=False)
        self.classifier = Mul(weight)

        self._initialize_weights()

    def forward(self, x):
        out = self.prep(x)
        out = self.res1(self.layer1(out))
        out = self.layer2(out)
        out = self.res3(self.layer3(out))

        out = self.pool(out).view(out.size(0), -1)
        out = self.classifier(self.linear(out))
        return F.log_softmax(out, dim=1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def finetune_parameters(self, iid, channels, weight, pool, **kw):
        self.linear = nn.Linear(channels['layer3'], self.new_num_classes, bias=False)
        self.classifier = Mul(weight)
        modules = [self.linear, self.classifier]
        for m in modules:
            for p in m.parameters():
                p.requires_grad = True
        return itertools.chain.from_iterable([m.parameters() for m in modules])


class ResNet9(nn.Module):
    def __init__(
        self, do_batchnorm=False, channels=None, weight=0.125, pool=nn.MaxPool2d(2),
        extra_layers=(), res_layers=('layer1', 'layer3'), width=1.0, **kw
    ):
        super().__init__()
        self.width = width
        # scale standard channels by width multiplier
        default_channels = channels or {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}
        self.channels = {k: max(1, int(v * width)) for k, v in default_channels.items()}
        # print(f"Using BatchNorm: {do_batchnorm}, Width Multiplier: {width}")
        self.n = BasicNet(do_batchnorm, self.channels, weight, pool, **kw)
        self.kw = kw

    def forward(self, x):
        return self.n(x)

    def finetune_parameters(self):
        return self.n.finetune_parameters(self.iid, self.channels, self.weight, self.pool, **self.kw)


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers with width scaling"""
    expansion = 4

    def __init__(
        self, in_channels, out_channels, stride=1, base_width=64, batch_norm=True, width=1.0
    ):
        super().__init__()
        self.batch_norm = batch_norm
        # compute internal channel width
        w = int(out_channels * (base_width / 64.0))
        layer_list = [nn.Conv2d(in_channels, w, kernel_size=1, bias=False)]
        if self.batch_norm:
            layer_list.append(nn.BatchNorm2d(w))
        layer_list += [nn.ReLU(inplace=True),
                       nn.Conv2d(w, w, stride=stride, kernel_size=3, padding=1, bias=False)]
        if self.batch_norm:
            layer_list.append(nn.BatchNorm2d(w))
        layer_list += [nn.ReLU(inplace=True),
                       nn.Conv2d(w, out_channels * BottleNeck.expansion, kernel_size=1, bias=False)]
        if self.batch_norm:
            layer_list.append(nn.BatchNorm2d(out_channels * BottleNeck.expansion))
        self.residual_function = nn.Sequential(*layer_list)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            sc_list = [nn.Conv2d(in_channels, out_channels * BottleNeck.expansion,
                                  stride=stride, kernel_size=1, bias=False)]
            if self.batch_norm:
                sc_list.append(nn.BatchNorm2d(out_channels * BottleNeck.expansion))
            self.shortcut = nn.Sequential(*sc_list)

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(
        self, block, num_block, base_width=64, num_classes=200,
        batch_norm=True, width=1.0
    ):
        super().__init__()
        self.width = width
        self.batch_norm = batch_norm
        # initial conv1 channels scaled
        self.in_channels = max(1, int(64 * width))
        if self.batch_norm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
            )
        # scale base_width for Bottleneck internals
        self.base_width = base_width * width
        # scaled layer channels
        self.conv2_x = self._make_layer(block, max(1, int(64 * width)), num_block[0], 1, self.base_width)
        self.conv3_x = self._make_layer(block, max(1, int(128 * width)), num_block[1], 2, self.base_width)
        self.conv4_x = self._make_layer(block, max(1, int(256 * width)), num_block[2], 2, self.base_width)
        self.conv5_x = self._make_layer(block, max(1, int(512 * width)), num_block[3], 2, self.base_width)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.in_channels already set by last _make_layer
        self.fc = nn.Linear(self.in_channels, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride, base_width):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s, base_width, self.batch_norm, width=self.width))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def _resnet(arch, block, num_block, base_width, num_classes, pretrained,
            batch_norm, width=1.0, model_dir="pretrained_models"):
    model = ResNet(block, num_block, base_width, num_classes, batch_norm, width)
    if pretrained:
        ckpt = torch.load(f"{model_dir}/{arch}-cifar{num_classes}.pt")
        state = ckpt.get("model_state_dict", ckpt)
        model_dict = model.state_dict()
        model_dict.update(state)
        model.load_state_dict(model_dict)
    return model


def ResNet50(class_num=200, pretrained=False, width=1.0, model_dir="pretrained_models"):
    return _resnet(
        "resnet50",
        BottleNeck,
        [3, 4, 6, 3],
        64,
        class_num,
        pretrained,
        batch_norm=True,
        width=width,
        model_dir=model_dir,
    )
