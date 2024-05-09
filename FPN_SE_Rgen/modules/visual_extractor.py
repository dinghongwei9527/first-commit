import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


# class VisualExtractor(nn.Module):
#     def __init__(self, args):
#         super(VisualExtractor, self).__init__()
#         self.visual_extractor = args.visual_extractor
#         self.pretrained = args.visual_extractor_pretrained
#         model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
#         modules = list(model.children())[:-2]
#         self.model = nn.Sequential(*modules)
#         self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
#
#     def forward(self, images):
#         patch_feats = self.model(images)
#         avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
#         batch_size, feat_size, _, _ = patch_feats.shape
#         patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
#         return patch_feats, avg_feats

class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        ).cuda()

    def forward(self, x):
        # 读取批数据图片数量及通道数
        b, c, h, w = x.size()
        # Fsq操作：经池化后输出b*c的矩阵
        y = self.gap(x).view(b, c)
        # Fex操作：经全连接层输出（b，c，1，1）矩阵
        y = self.fc(y).view(b, c, 1, 1)
        # Fscale操作：将得到的权重乘以原来的特征图x
        return x * y.expand_as(x)


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)

        self.layer0 = nn.Sequential(*list(model.children())[:4])
        # children layers
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # Top layer
        self.toplayer = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)
        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 2048, kernel_size=1, stride=1, padding=0)
        # 下采样
        self.downsample_conv = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.downsample_conv1 = nn.Conv2d(256, 256, kernel_size=1, stride=2)
        # 融合
        self.fusionlayer = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # 卷积
        self.conv = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                             nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
                             nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1))
        self.conv1 = nn.Conv2d(2048, 2048, kernel_size=1)
        # 变化
        self.trans = nn.Sequential(
            nn.Conv2d(256, 2048, kernel_size=1),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.trans_l = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 归一化
        self.bn1 = nn.BatchNorm2d(2048)

        # shortcut
        self.shortcut = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256))

        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        #print(model)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, images):
        # patch_feats = self.model(images)
        # avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        # batch_size, feat_size, _, _ = patch_feats.shape
        # patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        # return patch_feats, avg_feats
        # print('Images.shape:{}'.format(images.shape))
        c1 = self.layer0(images)
        # print('c1.shape:{}'.format(c1.shape))
        c2 = self.layer1(c1)
        # print('c2.shape:{}'.format(c2.shape))
        c3 = self.layer2(c2)
        # print('c3.shape:{}'.format(c3.shape))
        c4 = self.layer3(c3)
        # print('c4.shape:{}'.format(c4.shape))
        c5 = self.layer4(c4)
        # print('c5.shape:{}'.format(c5.shape))

        se = SE_Block(2048)
        p5 = self.toplayer(c5)
        p5 = se(p5)
        # print('p5.shape:{}'.format(p5.shape))
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p4 = se(p4)
        # print('p4.shape:{}'.format(p4.shape))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p3 = se(p3)
        # print('p3.shape:{}'.format(p3.shape))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p2 = se(p2)
        # print('p2.shape:{}'.format(p2.shape))

        # print('N:')
        # n2 = p2
        # print(n2.shape)
        # n3 = self.downsample_conv(n2) + p3
        # n3 = self.fusionlayer(n3)
        # n3 = F.relu(n3)
        # print(n3.shape)
        # n4 = self.downsample_conv(n3) + p4
        # n4 = self.fusionlayer(n4)
        # n4 = F.relu(n4)
        # print(n4.shape)
        # n5 = self.downsample_conv(n4) + p5
        # n5 = self.fusionlayer(n5)
        # n5 = self.shortcut(self.downsample_conv1(self.downsample_conv1(self.downsample_conv1(c2)))) + n5
        # n5 = F.relu(n5)
        # print(n5.shape)
        # patch_feats = self._upsample_add(n5, self.latlayer3(c2))
        # print(patch_feats.shape)
        # patch_feats = self.conv(patch_feats)
        #patch_feats = self.conv1(patch_feats)

        # patch_feats = se(patch_feats)
        fg = self.trans(c2) + se(self.trans(c2))
        fl = self.trans_l(p2) + se(self.trans_l(p2))
        # fg = self.trans(c2)
        # fl = self.trans_l(p2)
        patch_feats = fg + fl
        patch_feats = self.bn1(patch_feats)
        # print(patch_feats.shape)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        # print(patch_feats.shape)
        return patch_feats, avg_feats