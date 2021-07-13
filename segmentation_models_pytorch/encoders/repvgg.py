import torch.nn as nn
import numpy as np
import torch
from ..encoders._base import EncoderMixin

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            # print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)



#   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
#   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
#   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.detach().cpu(), bias.detach().cpu(),



class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self._temp_channels = list(map(lambda e, f: int(e * f), [64,128,256,512], width_multiplier))

        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1, deploy=self.deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)


    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

def create_RepVGG_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_A1(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_A2(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy)

def create_RepVGG_B0(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B1(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy)

def create_RepVGG_B1g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B1g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B2g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B2g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B3(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy)

def create_RepVGG_B3g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy)

def create_RepVGG_B3g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy)


func_dict = {
'RepVGG-A0': create_RepVGG_A0,
'RepVGG-A1': create_RepVGG_A1,
'RepVGG-A2': create_RepVGG_A2,
'RepVGG-B0': create_RepVGG_B0,
'RepVGG-B1': create_RepVGG_B1,
'RepVGG-B1g2': create_RepVGG_B1g2,
'RepVGG-B1g4': create_RepVGG_B1g4,
'RepVGG-B2': create_RepVGG_B2,
'RepVGG-B2g2': create_RepVGG_B2g2,
'RepVGG-B2g4': create_RepVGG_B2g4,
'RepVGG-B3': create_RepVGG_B3,
'RepVGG-B3g2': create_RepVGG_B3g2,
'RepVGG-B3g4': create_RepVGG_B3g4,
}
def get_RepVGG_func_by_name(name):
    return func_dict[name]


def whole_model_convert(train_model:torch.nn.Module, deploy_model:torch.nn.Module, save_path=None):
    all_weights = {}
    for name, module in train_model.named_modules():
        if hasattr(module, 'repvgg_convert'):
            kernel, bias = module.repvgg_convert()
            all_weights[name + '.rbr_reparam.weight'] = kernel
            all_weights[name + '.rbr_reparam.bias'] = bias
            print('convert RepVGG block')
        else:
            for p_name, p_tensor in module.named_parameters():
                full_name = name + '.' + p_name
                if full_name not in all_weights:
                    all_weights[full_name] = p_tensor.detach().cpu().numpy()
            for p_name, p_tensor in module.named_buffers():
                full_name = name + '.' + p_name
                if full_name not in all_weights:
                    all_weights[full_name] = p_tensor.cpu().numpy()

    deploy_model.load_state_dict(all_weights)
    if save_path is not None:
        torch.save(deploy_model.state_dict(), save_path)

    return deploy_model

def repvgg_model_convert(model:torch.nn.Module, build_func, save_path=None):
    converted_weights = {}
    for name, module in model.named_modules():
        if hasattr(module, 'repvgg_convert'):
            kernel, bias = module.repvgg_convert()
            converted_weights[name + '.rbr_reparam.weight'] = kernel
            converted_weights[name + '.rbr_reparam.bias'] = bias
        elif isinstance(module, torch.nn.Linear):
            converted_weights[name + '.weight'] = module.weight.detach().cpu().numpy()
            converted_weights[name + '.bias'] = module.bias.detach().cpu().numpy()
    del model

    deploy_model = build_func(deploy=True)
    for name, param in deploy_model.named_parameters():
        print('deploy param: ', name, param.size(), np.mean(converted_weights[name]))
        param.data = torch.from_numpy(converted_weights[name]).float()

    if save_path is not None:
        torch.save(deploy_model.state_dict(), save_path)

    return deploy_model


class RepVGGEncoder(RepVGG,EncoderMixin):
    def __init__(self,depth,**kwargs):

        super().__init__(**kwargs)

        self._in_channels = 3
        self._depth = depth
        self._out_channels = [self._in_channels,int(self._temp_channels[0])] + self._temp_channels
        del  self.linear,self.gap

    def get_stages(self):
        return [nn.Identity(),
                nn.Sequential(self.stage0),
                self.stage1,
                self.stage2,
                self.stage3,
                self.stage4
                ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth+1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        keys = list(state_dict.keys())
        for k in keys:
            if k.startswith("linear") :
                state_dict.pop(k)

        super().load_state_dict(state_dict, **kwargs)

url_map = {
    'repvgg_a0':'https://gitee.com/HX_98/RepVGG/attach_files/656683/download/RepVGG-A0-train.pth',
    'repvgg_a1':'https://gitee.com/HX_98/RepVGG/attach_files/656683/download/RepVGG-A1-train.pth',
    'repvgg_a2':'https://gitee.com/HX_98/RepVGG/attach_files/656683/download/RepVGG-A2-train.pth',
    'repvgg_b0':'https://gitee.com/HX_98/RepVGG/attach_files/656683/download/RepVGG-B0-train.pth',
    'repvgg_b1':'https://gitee.com/HX_98/RepVGG/attach_files/656683/download/RepVGG-B1-train.pth',
    'repvgg_b1g2':'https://gitee.com/HX_98/RepVGG/attach_files/656683/download/RepVGG-B1g2-train.pth',
    'repvgg_b1g4':'https://gitee.com/HX_98/RepVGG/attach_files/656683/download/RepVGG-B1g4-train.pth',
    'repvgg_b2':'https://gitee.com/HX_98/RepVGG/attach_files/656683/download/RepVGG-B2-train.pth',
    #############################
    'repvgg_b2g2':'https://gitee.com/HX_98/RepVGG/attach_files/656683/download/RepVGG-B2-train.pth',    #no model
    'repvgg_b2g4':'https://gitee.com/HX_98/RepVGG/attach_files/656683/download/RepVGG-B2g4-train.pth',
    'repvgg_b3':'https://gitee.com/HX_98/RepVGG/attach_files/656683/download/RepVGG-B3-200epochs-train.pth',
    'repvgg_b3g2':'https://gitee.com/HX_98/RepVGG/attach_files/656683/download/RepVGG-B2g4-train.pth', #no model
    'repvgg_b3g4':'https://gitee.com/HX_98/RepVGG/attach_files/656683/download/RepVGG-B3g4-200epochs-train.pth',
}

def _get_pretrained_settings(encoder):
    pretrained_settings = {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": url_map[encoder],
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    }
    return pretrained_settings

optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
# g2_map = {l: 2 for l in optional_groupwise_layers}
# g4_map = {l: 4 for l in optional_groupwise_layers}
repvgg_encoders = {
    "repvgg_a0": {
        "encoder": RepVGGEncoder,
        "pretrained_settings": _get_pretrained_settings("repvgg_a0"),
        "params": {
            "num_blocks":[2, 4, 14, 1],
            "width_multiplier":[0.75, 0.75, 0.75, 2.5],
            "override_groups_map":None,
        },
    },
    "repvgg_a1": {
        "encoder": RepVGGEncoder,
        "pretrained_settings": _get_pretrained_settings("repvgg_a1"),
        "params": {
            "num_blocks":[2, 4, 14, 1],
            "width_multiplier":[1, 1, 1, 2.5],
            "override_groups_map":None,
        },
    },
    "repvgg_a2": {
        "encoder": RepVGGEncoder,
        "pretrained_settings": _get_pretrained_settings("repvgg_a2"),
        "params": {
            "num_blocks":[2, 4, 14, 1],
            "width_multiplier":[1.5, 1.5, 1.5, 2.75],
            "override_groups_map":None,
        },
    },
    "repvgg_b0": {
        "encoder": RepVGGEncoder,
        "pretrained_settings": _get_pretrained_settings("repvgg_b0"),
        "params": {
            "num_blocks":[4, 6, 16, 1],
            "width_multiplier":[1, 1, 1, 2.5],
            "override_groups_map":None,
        },
    },
    "repvgg_b1": {
        "encoder": RepVGGEncoder,
        "pretrained_settings": _get_pretrained_settings("repvgg_b1"),
        "params": {
            "num_blocks":[4, 6, 16, 1],
            "width_multiplier":[2, 2, 2, 4],
            "override_groups_map":None,
        },
    },
    "repvgg_b1g2": {
        "encoder": RepVGGEncoder,
        "pretrained_settings": _get_pretrained_settings("repvgg_b1g2"),
        "params": {
            "num_blocks":[4, 6, 16, 1],
            "width_multiplier":[2, 2, 2, 4],
            "override_groups_map":g2_map,
        },
    },
    "repvgg_b1g4": {
        "encoder": RepVGGEncoder,
        "pretrained_settings": _get_pretrained_settings("repvgg_b1g4"),
        "params": {
            "num_blocks":[4, 6, 16, 1],
            "width_multiplier":[2, 2, 2, 4],
            "override_groups_map":g4_map,
        },
    },
    "repvgg_b2": {
        "encoder": RepVGGEncoder,
        "pretrained_settings": _get_pretrained_settings("repvgg_b2"),
        "params": {
            "num_blocks":[4, 6, 16, 1],
            "width_multiplier":[2.5, 2.5, 2.5, 5],
            "override_groups_map":None,
        },
    },
    "repvgg_b2g2": {
        "encoder": RepVGGEncoder,
        "pretrained_settings": _get_pretrained_settings("repvgg_b2g2"),
        "params": {
            "num_blocks":[4, 6, 16, 1],
            "width_multiplier":[2.5, 2.5, 2.5, 5],
            "override_groups_map":g2_map,
        },
    },
    "repvgg_b2g4": {
        "encoder": RepVGGEncoder,
        "pretrained_settings": _get_pretrained_settings("repvgg_b2g4"),
        "params": {
            "num_blocks":[4, 6, 16, 1],
            "width_multiplier":[2.5, 2.5, 2.5, 5],
            "override_groups_map":g4_map,
        },
    },
    "repvgg_b3": {
        "encoder": RepVGGEncoder,
        "pretrained_settings": _get_pretrained_settings("repvgg_b3"),
        "params": {
            "num_blocks":[4, 6, 16, 1],
            "width_multiplier":[3, 3, 3, 5],
            "override_groups_map":None,
        },
    },
    "repvgg_b3g2": {
        "encoder": RepVGGEncoder,
        "pretrained_settings": _get_pretrained_settings("repvgg_b3g2"),
        "params": {
            "num_blocks":[4, 6, 16, 1],
            "width_multiplier":[3, 3, 3, 5],
            "override_groups_map":g2_map,
        },
    },
    "repvgg_b3g4": {
        "encoder": RepVGGEncoder,
        "pretrained_settings": _get_pretrained_settings("repvgg_b3g4"),
        "params": {
            "num_blocks":[4, 6, 16, 1],
            "width_multiplier":[3, 3, 3, 5],
            "override_groups_map":g4_map,
        },
    },
}

if __name__ == '__main__':
    model = create_RepVGG_A0()
    print(model)
