import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from inplace_abn import InPlaceABN

# Output channel from layers given by the `extract_endpoints` function of 
# efficient net, use to initialize the fpn
output_feature_size = {
    0: [16, 24, 40, 112, 1280],
    1: [16, 24, 40, 112, 1280],
    2: [16, 24, 48, 120, 1408],
    3: [24, 32, 48, 136, 1536],
    4: [24, 32, 56, 160, 1792],
    5: [24, 40, 64, 176, 2048],
    6: [32, 40, 72, 200, 2304],
    7: [32, 48, 80, 224, 2560],
    8: [32, 56, 88, 248, 2816]
}

def generate_backbone_EfficientPS(id_effi_net):

    backbone = EfficientNet.from_name('efficientnet-b{}'.format(id_effi_net))
    
    backbone._bn0 = InPlaceABN(num_features=backbone._bn0.num_features)
    backbone._bn1 = InPlaceABN(num_features=backbone._bn1.num_features)
    backbone._swish = nn.Identity()
    for block in backbone._blocks:
        # Remove SE block
        block.has_se = False

        # Replace BN with Inplace BN (default activation is leaky relu)
        block._bn1 = InPlaceABN(num_features=block._bn1.num_features)
        block._bn2 = InPlaceABN(num_features=block._bn2.num_features)

        # Remove swish activation since Inplace BN contains the activation layer
        block._swish = nn.Identity()

    return backbone