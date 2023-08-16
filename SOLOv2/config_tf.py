import json
from pathlib import Path
import os


class Config_tf():

    def __init__(self, **kwargs):
        self.load_pretrained_tf = True
        self.load_backbone = False
        self.backbone_params = {}
        self.backbone = "tf.keras.applications.resnet.ResNet101"

        # Specific params
        self.ncls = 1
        self.imshape = (1024, 1024, 3)

        # General layers params
        self.activation = 'gelu'
        self.normalization = "gn"  # gn for Group Norm
        self.normalization_kw = {'groups': 32}
        self.model_name = "SOLOv2-Resnext101"

        # FPN
        self.connection_layers = {"C1": "conv1_relu",
                                 "C2": "conv2_block3_out",
                                  "C3": "conv3_block4_out",
                                  "C4": "conv4_block23_out",
                                  "C5": "conv5_block3_out"}
        self.FPN_filters = 256
        self.extra_FPN_layers = 1  # layers after P5. Strides must correspond to the number of FPN layers !

        # SOLO head
        self.head_filters = [256, 256, 256, 256]  # Filters per stage
        self.strides = [4, 8, 16, 32, 64]
        self.head_layers = 4  # Number of repeats of head conv layers
        self.head_filters = 256
        self.kernel_size = 1
        self.grid_sizes = [64, 36, 24, 16, 12]
        self.scale_ranges = [[1, 96], [48, 192], [96, 384], [192, 768], [384, 2048]]
        self.offset_factor = 0.25

        # SOLO MASK head
        self.mask_mid_filters = 128
        self.mask_output_filters = 256

        # loss
        self.lossweights = [1., 1.]

        # Update defaults parameters with kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):

        s = ""

        for k, v in self.__dict__.items():
            s += "{}:{}\n".format(k, v)

        return s

    def save(self, filename):

        # data = {k:v for k, v in self.__dict__.items()}

        p = Path(filename).parent.absolute()
        if not os.path.isdir(p):
            os.mkdir(p)

        with open(filename, 'w') as f:
            json.dump(self.__dict__, f)
