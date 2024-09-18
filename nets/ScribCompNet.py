import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BN_MOMENTUM, hrnet_classification, BasicBlock


class HRnet_Backbone(nn.Module):
    def __init__(self, backbone = 'hrnetv2_w18', pretrained = False):
        super(HRnet_Backbone, self).__init__()
        self.model    = hrnet_classification(backbone = backbone, pretrained = pretrained)
        del self.model.incre_modules
        del self.model.downsamp_modules
        del self.model.final_layer
        del self.model.classifier
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)
        
        x_list = []
        for i in range(2):
            if self.model.transition1[i] is not None:
                x_list.append(self.model.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.model.stage2(x_list)

        x_list = []
        for i in range(3):
            if self.model.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.model.transition2[i](y_list[i]))
                else:
                    x_list.append(self.model.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage3(x_list)

        x_list = []
        for i in range(4):
            if self.model.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.model.transition3[i](y_list[i]))
                else:
                    x_list.append(self.model.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage4(x_list)
        
        return y_list

class ScribCompNet(nn.Module):
    def __init__(self, num_classes = 21, backbone = 'hrnetv2_w18', pretrained = False):
        super(ScribCompNet, self).__init__()
        self.backbone       = HRnet_Backbone(backbone = backbone, pretrained = pretrained)

        last_inp_channels   = np.int(np.sum(self.backbone.model.pre_stage_channels))
        last_inp_channels1  = 64

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        self.final_layer = nn.Conv2d(in_channels=last_inp_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0)

        self.last_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels1, out_channels=last_inp_channels1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels1, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=last_inp_channels1, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

        self.deconv_layers = self._make_deconv_layers(last_inp_channels+num_classes, last_inp_channels1)

    def _make_deconv_layers(self, input_channels, output_channels):
        deconv_layers = []
        layers = []
        layers.append(nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=input_channels,     # 18+36+7
                out_channels=output_channels,   # 32
                kernel_size=4,     # 4
                stride=2,
                padding=1,         # 1
                output_padding=0,  # 0
                bias=False),
            nn.BatchNorm2d(output_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        ))
        
        for _ in range(2): 
            layers.append(nn.Sequential(
                BasicBlock(output_channels, output_channels),
            ))
        deconv_layers.append(nn.Sequential(*layers))
        input_channels = output_channels

        return nn.ModuleList(deconv_layers)
    
    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone(inputs)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x = torch.cat([x[0], x1, x2, x3], 1)
        
        x = self.last_layer(x) # 270*120*120
        aux_fea = self.final_layer(x) # 7*120*120
        aux_seg = F.interpolate(aux_fea, size=(H, W), mode='bilinear', align_corners=True)

        y = torch.cat([aux_fea, x], 1)
        y = self.deconv_layers[0](y)
        y = self.last_layer_1(y)
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)

        return y, aux_seg