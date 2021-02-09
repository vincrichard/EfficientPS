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

def generate_backbone_EfficientPS(cfg):
    """
    Create an EfficientNet model base on this repository:
    https://github.com/lukemelas/EfficientNet-PyTorch

    Modify the existing Efficientnet base on the EfficientPS paper,
    ie:
    - replace BN and swish with InplaceBN and LeakyRelu
    - remove se (squeeze and excite) blocks
    Args:
    - cdg (Config) : config object
    Return:
    - backbone (nn.Module) : Modify version of the EfficentNet
    """

    if cfg.MODEL_CUSTOM.BACKBONE.LOAD_PRETRAIN:
        backbone = EfficientNet.from_pretrained(
            'efficientnet-b{}'.format(cfg.MODEL_CUSTOM.BACKBONE.EFFICIENTNET_ID))
    else:
        backbone = EfficientNet.from_name(
            'efficientnet-b{}'.format(cfg.MODEL_CUSTOM.BACKBONE.EFFICIENTNET_ID))

    backbone._bn0 = InPlaceABN(num_features=backbone._bn0.num_features, eps=0.001)
    backbone._bn1 = InPlaceABN(num_features=backbone._bn1.num_features, eps=0.001)
    backbone._swish = nn.Identity()
    for i, block in enumerate(backbone._blocks):
        # Remove SE block
        block.has_se = False
        # Additional step to have the correct number of parameter on compute
        block._se_reduce =  nn.Identity()
        block._se_expand = nn.Identity()
        # Replace BN with Inplace BN (default activation is leaky relu)
        if '_bn0' in [name for name, layer in block.named_children()]:
            block._bn0 = InPlaceABN(num_features=block._bn0.num_features, eps=0.001)
        block._bn1 = InPlaceABN(num_features=block._bn1.num_features, eps=0.001)
        block._bn2 = InPlaceABN(num_features=block._bn2.num_features, eps=0.001)

        # Remove swish activation since Inplace BN contains the activation layer
        block._swish = nn.Identity()

    return backbone