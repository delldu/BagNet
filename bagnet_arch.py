import math
import torch.nn as nn
from torch.utils import model_zoo
import matplotlib.pyplot as plt
import numpy as np

import pdb

model_urls = {
    'bagnet9':
    'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet8-34f4ccd2.pth.tar',
    'bagnet17':
    'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet16-105524de.pth.tar',
    'bagnet33':
    'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet32-2ddd53ed.pth.tar',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 kernel_size=1):
        super(Bottleneck, self).__init__()
        # print('Creating bottleneck with kernel size {} and stride {} with padding {}'.format(kernel_size, stride, (kernel_size - 1) // 2))
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False)  # changed padding from (kernel_size - 1) // 2
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


      # (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      # (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      # (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), bias=False)
      # (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      # (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      # (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      # (relu): ReLU(inplace=True)
      # (downsample): Sequential(
      #   (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
      #   (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      # )



    def forward(self, x, **kwargs):
        # pdb.set_trace()
        # (Pdb) pp x.size()
        # torch.Size([1, 64, 62, 62])

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if residual.size(-1) != out.size(-1):
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:, :, :-diff, :-diff]

        out += residual
        out = self.relu(out)
        # pdb.set_trace()
        # torch.Size([1, 256, 30, 30])

        return out


class BagNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 strides=[1, 2, 2, 2],
                 kernel3=[0, 0, 0, 0],
                 num_classes=1000,
                 avg_pool=True):
        self.inplanes = 64
        super(BagNet, self).__init__()
        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(64,
                               64,
                               kernel_size=3,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block,
                                       64,
                                       layers[0],
                                       stride=strides[0],
                                       kernel3=kernel3[0],
                                       prefix='layer1')
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=strides[1],
                                       kernel3=kernel3[1],
                                       prefix='layer2')
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=strides[2],
                                       kernel3=kernel3[2],
                                       prefix='layer3')
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=strides[3],
                                       kernel3=kernel3[3],
                                       prefix='layer4')
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.avg_pool = avg_pool
        self.block = block

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # pdb.set_trace()
                # (Pdb) pp m.weight.std()
                # tensor(0.1823, grad_fn=<StdBackward0>)
                # (Pdb) pp math.sqrt(2. / n)
                # 0.1767766952966369

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    kernel3=0,
                    prefix=''):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        # (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), bias=False)
        # (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # (relu): ReLU(inplace=True)
        # (downsample): Sequential(
        #   (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        #   (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # )

        # (1): Bottleneck(
        #   (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #   (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (conv2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #   (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        #   (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #   (relu): ReLU(inplace=True)
        # )

        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample,
                  kernel_size=kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Wouldn't it be great if we had pipes?
        # print("xxxx --- x.size()=", x.size())
        # pdb.set_trace()
        # xxxx --- x.size()= torch.Size([64, 3, 28, 28])

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x.size() = torch.Size([64, 64, 26, 26])
        x = self.layer1(x)
        # x.size() = torch.Size([64, 64, 12, 12])
        x = self.layer2(x)
        # x.size() = torch.Size([64, 64, 5, 5])
        x = self.layer3(x)
        # x.size() = torch.Size([64, 64, 2, 2])
        x = self.layer4(x)
        # x.size() = torch.Size([64, 64, 2, 2])

        # print("pre pool", x.shape)
        # yyyy --- x.size()= torch.Size([64, 2048, 3, 3])
        # yyyy --- x.size()= torch.Size([64, 2048, 2, 2])

        if self.avg_pool:
            x = nn.AvgPool2d(x.size()[2], stride=1)(x)
            # print("post pool", x.shape)
            # post pool torch.Size([64, 2048, 1, 1])

            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.fc(x)

        # pdb.set_trace()
        # (fc): Linear(in_features=2048, out_features=1000, bias=True)

        return x


def create_bagnet33(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-33 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(Bottleneck, [3, 4, 6, 3],
                   strides=strides,
                   kernel3=[1, 1, 1, 1],
                   **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet33']))
    return model


def create_bagnet17(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-17 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(Bottleneck, [3, 4, 6, 3],
                   strides=strides,
                   kernel3=[1, 1, 1, 0],
                   **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet17']))
    return model


def create_bagnet9(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-9 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(Bottleneck, [3, 4, 6, 3],
                   strides=strides,
                   kernel3=[1, 1, 0, 0],
                   **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet9']))
    return model


def generate_heatmap(model, image, target, patchsize):
    with torch.no_grad():
        # pad with zeros
        _, c, x, y = image.shape
        padded_image = np.zeros((c, x + patchsize - 1, y + patchsize - 1))
        padded_image[:, (patchsize - 1) // 2:(patchsize - 1) // 2 +
                     x, (patchsize - 1) // 2:(patchsize - 1) // 2 +
                     y] = image[0]
        image = padded_image[None].astype(np.float32)

        # extract patches
        input = torch.from_numpy(image).cuda()
        patches = input.permute(0, 2, 3, 1)
        patches = patches.unfold(1, patchsize, 1).unfold(2, patchsize, 1)
        num_rows = patches.shape[1]
        num_cols = patches.shape[2]
        patches = patches.contiguous().view((-1, 3, patchsize, patchsize))

        # compute logits for each patch
        logits_list = []
        for batch_patches in torch.split(patches, 1000):
            logits = model(batch_patches)
            logits = logits[:, target]
            logits_list.append(logits.data.cpu().numpy().copy())

        logits = np.hstack(logits_list)
        return logits.reshape((28, 28))

if __name__ == '__main__':
    import torch
    from fastai.datasets import URLs, untar_data
    from fastai.vision import ImageDataBunch, cnn_learner
    # from fastai.vision import create_cnn
    from fastai.metrics import accuracy

    path = untar_data(URLs.MNIST_SAMPLE)
    mnist = ImageDataBunch.from_folder(path)

    # pdb.set_trace()
    # PosixPath('/home/dell/.fastai/data/mnist_sample')

    # bagnet = create_bagnet9(num_classes=len(mnist.classes))
    bagnet = create_bagnet9(num_classes=len(mnist.classes)).cuda()
    print(bagnet)

    bagnet.load_state_dict(torch.load("bagnet.pth"))

    # learner = cnn_learner(
    #     data=mnist,
    #     # mmkay I definitely wouldn't've guessed this
    #     base_arch=lambda _: bagnet,
    #     metrics=accuracy
    # )
    # learner.fit_one_cycle(cyc_len=10, max_lr=1e-2)

    # torch.save(bagnet.state_dict(), "bagnet.pth")

    images, labels = mnist.one_batch()
    i = 1
    img = images[i].cpu().numpy()
    plt.imshow(np.rollaxis(img, 0, 3))
    plt.show()

    heatmap = generate_heatmap(bagnet, img[None], labels[i].cpu().numpy().item(), 9)
    plt.imshow(heatmap, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.show()




    # x = torch.rand(64, 3, 28, 28)
    # bagnet(x)

    # bagnet = bagnet.cuda()

    # learner = cnn_learner(
    #     data=mnist,
    #     # mmkay I definitely wouldn't've guessed this
    #     base_arch=lambda _: bagnet,
    #     metrics=accuracy
    # )
    # # model = create_bagnet9(pretrained=True)
    # # print(model)
    # learner.fit_one_cycle(cyc_len=2, max_lr=1e-2)
