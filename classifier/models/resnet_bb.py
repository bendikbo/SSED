import math
from collections.abc import Iterable
import torch
import torch.nn as nn

def _quick_assert_cfg(cfg, function_name, attribute_dict):
    """
    Function to assert all config attributes required for another function
    Input: config from function to be asserted,
    dict with attribute name mapped either to viable type or list of viable types
    name of function to be asserted.
    """
    for attribute in attribute_dict:
        assert hasattr(
            cfg,
            str(attribute)), """
            Function:
            {}
            requires cfg object with attribute:
            {}
            """.format(function_name, attribute)
        if isinstance(attribute_dict[attribute], Iterable):
            one_of_type_options = False
            attribute_type = str
            for type_option in attribute_dict[attribute]:
                if isinstance(cfg.__getattr__(attribute), type_option):
                    one_of_type_options = True
                else:
                    attribute_type = type(cfg.__getattr__(attribute))
            assert one_of_type_options, """
            Function: 
            {} 
            expected attribute: 
            {} 
            of cfg object to be type: 
            {}
            got:
            {}
            """.format(attribute, function_name, type_option, attribute_type)
        else:
            assert isinstance(cfg.__getattr__(attribute),
                attribute_dict[attribute]),"""
            Function: 
            {} 
            expected attribute:
            {} 
            of cfg object to be type: 
            {}
            got:
            {}
            """.format(function_name, attribute,attribute_dict[attribute], type(cfg.__getattr__(attribute)))


class InceptBlock1(nn.Module):
    """
    Class for the first block of the model
    First block is 4 parallel conv-layers with differing kernel sizes
    Just like a naive inception block
    """
    def __init__(self, cfg):
        super(InceptBlock1, self).__init__()
        attribute_dict = {
            "OUT_CH" : int
        }
        _quick_assert_cfg(cfg, "InceptResBlock1", attribute_dict)
        parallel_output_channels = math.floor(cfg.OUT_CH/4)
        #Parallel blocks to be depth-concatenated
        #(IN_CH, IN_RES) = (1,262144) (May be outdated comment)
        #OUT = (OUT_CH, IN_RES/64)
        self.conv_p1 = nn.Sequential(
            nn.Conv1d(
                in_channels = 1,
                out_channels = parallel_output_channels,
                kernel_size = 33,
                stride = 4,
                padding = 16
                ),#(OUT_CH/4, IN/4)
            nn.MaxPool1d(
                kernel_size = 17,
                stride = 16,
                padding = 8
            )#(OUT_CH/4, IN/64)
        )
        self.conv_p2 = nn.Sequential(
            nn.Conv1d(
                in_channels = 1,
                out_channels = parallel_output_channels,
                kernel_size = 17,
                stride = 4,
                padding = 8
                ),#(OUT_CH/4, IN_RES/4)
            nn.MaxPool1d(
                kernel_size = 17,
                stride = 16,
                padding = 8
            )#(OUT_CH/4, IN_RES/64)
        )
        self.conv_p3 = nn.Sequential(
            nn.Conv1d(
                in_channels = 1,
                out_channels = parallel_output_channels,
                kernel_size = 9,
                stride = 4,
                padding = 4
                ),#(OUT_CH/4, IN_RES/4)
            nn.MaxPool1d(
                kernel_size = 17,
                stride = 16,
                padding = 8
            )#(OUT_CH/4, IN_RES/64)
        )
        self.conv_p4 = nn.Sequential(
            nn.Conv1d(
                in_channels = 1,
                out_channels = parallel_output_channels,
                kernel_size = 5,
                stride = 4,
                padding = 2
                ),#(OUT_CH/4, IN_RES/4)
            nn.MaxPool1d(
                kernel_size = 17,
                stride = 16,
                padding = 8
            )#(OUT_CH/4, IN_RES/64)
        )
    def forward(self, x):
        """
        Forward function does depth-wise concatenation with the
        different convolutions.
        """
        x = torch.cat(
                (self.conv_p1(x),
                self.conv_p2(x),
                self.conv_p3(x),
                self.conv_p4(x)),
                dim = 1)#(OUT_CH, IN_RES/64)
        return x


class InceptBlock2(nn.Module):
    """
    Class for the first block of the model
    First block is N parallel conv-layers with differing kernel sizes
    Just like a naive inception block
    """
    def __init__(self, cfg):
        super(InceptBlock2, self).__init__()
        attribute_dict = {
            "OUT_CH" : int
        }
        _quick_assert_cfg(cfg, "InceptResBlock1", attribute_dict)
        parallel_output_channels = math.floor(cfg.OUT_CH/4)
        #Parallel blocks to be depth-concatenated
        #(IN_CH, IN_RES) = (1,262144) (May be outdated comment)
        #OUT = (OUT_CH, IN_RES/64)
        self.conv_p1 = nn.Sequential(
            nn.Conv1d(
                in_channels = 1,
                out_channels = parallel_output_channels,
                kernel_size = 33,
                stride = 4,
                padding = 16
                ),
                nn.LeakyReLU(),
        )
        self.conv_p2 = nn.Sequential(
            nn.Conv1d(
                in_channels = 1,
                out_channels = parallel_output_channels,
                kernel_size = 17,
                stride = 4,
                padding = 8
                ),
                nn.LeakyReLU()
        )
        self.conv_p3 = nn.Sequential(
            nn.Conv1d(
                in_channels = 1,
                out_channels = parallel_output_channels,
                kernel_size = 9,
                stride = 4,
                padding = 4
                ),
                nn.LeakyReLU()
        )
        self.conv_p4 = nn.Sequential(
            nn.Conv1d(
                in_channels = 1,
                out_channels = parallel_output_channels,
                kernel_size = 5,
                stride = 4,
                padding = 2
                ),
                nn.LeakyReLU()
        )
        self.gauge_conv = nn.Sequential(
            nn.Conv1d(
                in_channels = cfg.OUT_CH,
                out_channels = cfg.OUT_CH,
                kernel_size = 65,
                stride = 16,
                padding = 32
            )
        )
    def forward(self, x):
        """
        Forward function does depth-wise concatenation with the
        different convolutions.
        """
        x = torch.cat(
                (self.conv_p1(x),
                self.conv_p2(x),
                self.conv_p3(x),
                self.conv_p4(x)),
                dim = 1)#(OUT_CH, IN_RES/4)
        x = self.gauge_conv(x)
        return x

class Resnet18Block1D(nn.Module):
    """
    Class for a basic Resnet 9/18/34 block,
    will automatically add downsampling shortcut
    as in resnet paper if there is resolution mismatch.
    """
    def __init__(self, cfg):
        super(Resnet18Block1D, self).__init__()
        attribute_dict = {
            "IN_CH"     : int,
            "OUT_CH"    : int,
            "KERNEL"    : int,
            "PADDING"   : int
        }
        _quick_assert_cfg(cfg, "Resnet18Block1D", attribute_dict)
        if not hasattr(cfg, "STRIDE"):
            self.stride = 1
        else:
            self.stride = cfg.STRIDE
        if hasattr(cfg, "RELU"):
            if cfg.RELU == "leaky":
                self.relu = nn.LeakyReLU()
            else:
                self.relu = nn.ReLU()
        else:
            self.relu = nn.ReLU()
        self.main_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=cfg.IN_CH,
                out_channels=cfg.OUT_CH,
                kernel_size=cfg.KERNEL,
                stride=self.stride,
                padding=cfg.PADDING
            ),
            self.relu,
            nn.BatchNorm1d(cfg.OUT_CH),
            nn.Conv1d(
                in_channels=cfg.OUT_CH,
                out_channels=cfg.OUT_CH,
                kernel_size=cfg.KERNEL,
                stride=1,
                padding=cfg.PADDING
            )
        )
        if cfg.IN_CH != cfg.OUT_CH or self.stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels=cfg.IN_CH,
                    out_channels=cfg.OUT_CH,
                    kernel_size=1,
                    stride=self.stride,
                    padding=0
                ),
                nn.BatchNorm1d(cfg.OUT_CH)
            )
        else:
            self.shortcut = None
    def forward(self, x):
        residual = x
        x = self.main_conv(x)
        if self.shortcut is not None:
            residual = self.shortcut(residual)
            x += residual
        else:
            x += residual
        x = self.relu(x)
        return x


class Resnet50Block1D(nn.Module):
    """
    Class for typical resnet50-block,
    in, mid and out channels are all specified due to
    change between the bigger convolutional layers
    """
    def __init__(self, cfg):
        super(Resnet50Block1D, self).__init__()
        attribute_dict = {
            "IN_CH"     : int,
            "MID_CH"    : int,
            "OUT_CH"    : int,
            "STRIDE"    : int,
            "KERNEL"    : int,
            "PADDING"   : int
        }
        _quick_assert_cfg(cfg, "ResnetBlock1D", attribute_dict)
        if hasattr(cfg, "RELU"):
            if cfg.RELU == "leaky":
                self.relu = nn.LeakyReLU()
            else:
                self.relu = nn.ReLU()
        else:
            self.relu = nn.ReLU()
        self.main_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=cfg.IN_CH,
                out_channels=cfg.MID_CH,
                kernel_size = 1,
                stride=1,
            ),
            self.relu,
            nn.BatchNorm1d(cfg.MID_CH),
            nn.Conv1d(
                in_channels=cfg.MID_CH,
                out_channels=cfg.MID_CH,
                kernel_size=cfg.KERNEL,
                padding=cfg.PADDING,
                stride=cfg.STRIDE
            ),
            self.relu,
            nn.BatchNorm1d(cfg.MID_CH),
            nn.Conv1d(
                in_channels=cfg.MID_CH,
                out_channels=cfg.OUT_CH,
                kernel_size=1,
                stride=1
            )
        )
        if cfg.IN_CH != cfg.OUT_CH or cfg.STRIDE != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels=cfg.IN_CH,
                    out_channels=cfg.OUT_CH,
                    kernel_size=1,
                    stride=cfg.STRIDE,
                    padding=0
                ),
                nn.BatchNorm1d(cfg.OUT_CH)
            )
        else:
            self.shortcut = None
    def forward(self, x):
        residual = x
        x = self.main_conv(x)
        if self.shortcut is not None:
            residual = self.shortcut(residual)
            x += residual
        else:
            x += residual
        x = self.relu(x)
        return x


class ResnetAvgPool(nn.Module):
    """
    Class to create average pool at the end of the Resnet.
    Will create average pool with set kernel and stride.
    """
    def __init__(self, cfg):
        super(ResnetAvgPool, self).__init__()
        attribute_dict = {
            "KERNEL"    :   int,
            "STRIDE"    :   int
        }
        _quick_assert_cfg(cfg, "ResnetAvgPool init", attribute_dict)
        self.avg_pool = nn.AvgPool1d(
            kernel_size=cfg.KERNEL,
            stride=cfg.STRIDE
        )
    def forward(self, x):
        x = self.avg_pool(x)
        return x


class Resnet501D(nn.Module):
    """
    Main Resnet-class, Optional classifier is dropped by setting
    classifier to false in the class initialization.
    Configuration object needs to have correct parameters for all 
    layers, read up on the Resnet50Block1D class to understand.
    """
    def __init__(self, cfg, classifier = True):
        super(Resnet501D, self).__init__()
        self.classifier = classifier
        self.conv1 = InceptBlock1(cfg.CONV1)
        self.conv2_1 = Resnet50Block1D(cfg.CONV2_1)
        self.conv2_2 = Resnet50Block1D(cfg.CONV2_2)
        self.conv2_3 = Resnet50Block1D(cfg.CONV2_3)
        self.conv3_1 = Resnet50Block1D(cfg.CONV3_1)
        self.conv3_2 = Resnet50Block1D(cfg.CONV3_2)
        self.conv3_3 = Resnet50Block1D(cfg.CONV3_3)
        self.conv3_4 = Resnet50Block1D(cfg.CONV3_4)
        self.conv4_1 = Resnet50Block1D(cfg.CONV4_1)
        self.conv4_2 = Resnet50Block1D(cfg.CONV4_2)
        self.conv4_3 = Resnet50Block1D(cfg.CONV4_3)
        self.conv4_4 = Resnet50Block1D(cfg.CONV4_4)
        self.conv4_5 = Resnet50Block1D(cfg.CONV4_5)
        self.conv4_6 = Resnet50Block1D(cfg.CONV4_6)
        self.conv5_1 = Resnet50Block1D(cfg.CONV5_1)
        self.conv5_2 = Resnet50Block1D(cfg.CONV5_2)
        self.conv5_3 = Resnet50Block1D(cfg.CONV5_3)
        self.avg_pool = ResnetAvgPool(cfg.AVG_POOL)
        if classifier:
            self.full_conn = nn.Sequential(
                nn.Linear(
                    in_features=cfg.CONV5_3.OUT_CH,
                    out_features=cfg.CONV5_3.OUT_CH
                ),
                nn.Linear(
                    in_features=cfg.CONV5_3.OUT_CH,
                    out_features=cfg.CLASSIFIER.NUM_CLASSES
                )
            )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.avg_pool(x)
        if self.classifier:
            x = torch.squeeze(x)#DIM should be (1, CONF5_3.OUT_CH)
            x = self.full_conn(x)
        return x


class BOGNet(nn.Module):
    """
    Main BOGNet-class, Optional classifier is dropped by setting
    classifier to false in the class initialization.
    Configuration object needs to have correct parameters for all 
    layers, read up on the Resnet50Block1D class to understand.
    """
    def __init__(self, cfg, classifier = True):
        super(BOGNet, self).__init__()
        self.classifier = classifier
        self.conv1 = InceptBlock2(cfg.CONV1)
        self.conv2_1 = Resnet50Block1D(cfg.CONV2_1)
        self.conv2_2 = Resnet50Block1D(cfg.CONV2_2)
        self.conv2_3 = Resnet50Block1D(cfg.CONV2_3)
        self.conv3_1 = Resnet50Block1D(cfg.CONV3_1)
        self.conv3_2 = Resnet50Block1D(cfg.CONV3_2)
        self.conv3_3 = Resnet50Block1D(cfg.CONV3_3)
        self.conv3_4 = Resnet50Block1D(cfg.CONV3_4)
        self.conv4_1 = Resnet50Block1D(cfg.CONV4_1)
        self.conv4_2 = Resnet50Block1D(cfg.CONV4_2)
        self.conv4_3 = Resnet50Block1D(cfg.CONV4_3)
        self.conv4_4 = Resnet50Block1D(cfg.CONV4_4)
        self.conv4_5 = Resnet50Block1D(cfg.CONV4_5)
        self.conv4_6 = Resnet50Block1D(cfg.CONV4_6)
        self.conv5_1 = Resnet50Block1D(cfg.CONV5_1)
        self.conv5_2 = Resnet50Block1D(cfg.CONV5_2)
        self.conv5_3 = Resnet50Block1D(cfg.CONV5_3)
        self.avg_pool = ResnetAvgPool(cfg.AVG_POOL)
        if classifier:
            self.full_conn = nn.Sequential(
                nn.Linear(
                    in_features=cfg.CONV5_3.OUT_CH,
                    out_features=cfg.CONV5_3.OUT_CH
                ),
                nn.Linear(
                    in_features=cfg.CONV5_3.OUT_CH,
                    out_features=cfg.CLASSIFIER.NUM_CLASSES
                )
            )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.avg_pool(x)
        if self.classifier:
            x = torch.squeeze(x)#DIM should be (1, CONF5_3.OUT_CH)
            x = self.full_conn(x)
        return x

class Resnet91D(nn.Module):
    """
    1 Dimensional 9 layer resnet module.
    Had to be made due to little training time.
    """
    def __init__(self, cfg, classifier = True):
        super(Resnet91D, self).__init__()
        self.classifier = classifier
        self.conv1 = InceptBlock1(cfg.CONV1)
        self.conv2 = Resnet18Block1D(cfg.CONV2)
        self.conv3 = Resnet18Block1D(cfg.CONV3)
        self.conv4 = Resnet18Block1D(cfg.CONV4)
        self.conv5 = Resnet18Block1D(cfg.CONV5)
        self.avg_pool = ResnetAvgPool(cfg.AVG_POOL)
        if classifier:
            self.full_conn = nn.Sequential(
                nn.Linear(
                    in_features=cfg.CONV5.OUT_CH,
                    out_features=cfg.CONV5.OUT_CH
                ),
                nn.Linear(
                    in_features=cfg.CONV5.OUT_CH,
                    out_features=cfg.CLASSIFIER.NUM_CLASSES
                )
            )
    def forward(self, x):
        x = self.conv1(x)#(B, C1_O, )
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        if self.classifier:
            x = torch.squeeze(x)#dim should be (batch_size, CONV5.OUT_CH)
            x = self.full_conn(x)
        return x


class Resnet341D(nn.Module):
    """
    Class for a 34 layer one dimensional Residual Neural Network
    With a naïve inception module at the input layer.
    """
    def __init__(self, cfg, classifier = True):
        super(Resnet341D, self).__init__()
        self.classifier = classifier
        self.conv1 = InceptBlock1(cfg.CONV1)
        self.conv2_1 = Resnet18Block1D(cfg.CONV2_1)
        self.conv2_2 = Resnet18Block1D(cfg.CONV2_2)
        self.conv2_3 = Resnet18Block1D(cfg.CONV2_3)
        self.conv3_1 = Resnet18Block1D(cfg.CONV3_1)
        self.conv3_2 = Resnet18Block1D(cfg.CONV3_2)
        self.conv3_3 = Resnet18Block1D(cfg.CONV3_3)
        self.conv3_4 = Resnet18Block1D(cfg.CONV3_4)
        self.conv4_1 = Resnet18Block1D(cfg.CONV4_1)
        self.conv4_2 = Resnet18Block1D(cfg.CONV4_2)
        self.conv4_3 = Resnet18Block1D(cfg.CONV4_3)
        self.conv4_4 = Resnet18Block1D(cfg.CONV4_4)
        self.conv4_5 = Resnet18Block1D(cfg.CONV4_5)
        self.conv4_6 = Resnet18Block1D(cfg.CONV4_6)
        self.conv5_1 = Resnet18Block1D(cfg.CONV5_1)
        self.conv5_2 = Resnet18Block1D(cfg.CONV5_2)
        self.conv5_3 = Resnet18Block1D(cfg.CONV5_3)
        self.avg_pool = ResnetAvgPool(cfg.AVG_POOL)
        if classifier:
            self.full_conn = nn.Sequential(
                nn.Linear(
                    in_features=cfg.CONV5_3.OUT_CH,
                    out_features=cfg.CONV5_3.OUT_CH
                ),
                nn.Linear(
                    in_features=cfg.CONV5_3.OUT_CH,
                    out_features=cfg.CLASSIFIER.NUM_CLASSES
                )
            )
    def forward(self, x):
        #print(x.size())
        x = self.conv1(x)
        #print(x.size())	
        x = self.conv2_1(x)
        #print(x.size())
        x = self.conv2_2(x)
        #print(x.size())
        x = self.conv2_3(x)
        #print(x.size())
        x = self.conv3_1(x)
        #print(x.size())
        x = self.conv3_2(x)
        #print(x.size())
        x = self.conv3_3(x)
        #print(x.size())
        x = self.conv3_4(x)
        #print(x.size())
        x = self.conv4_1(x)
        #print(x.size())
        x = self.conv4_2(x)
        #print(x.size())
        x = self.conv4_3(x)
        #print(x.size())
        x = self.conv4_4(x)
        #print(x.size())
        x = self.conv4_5(x)
        #print(x.size())
        x = self.conv4_6(x)
        #print(x.size())
        x = self.conv5_1(x)
        #print(x.size())
        x = self.conv5_2(x)
        #print(x.size())
        x = self.conv5_3(x)
        #print(x.size())
        x = self.avg_pool(x)
        if self.classifier:
            x = torch.squeeze(x)#DIM should be (1, CONF5_3.OUT_CH)
            x = self.full_conn(x)
        return x
