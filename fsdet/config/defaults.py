# 引入detectron2中默认config
from detectron2.config.defaults import _C

### IMPORTANT：基于detectron2的默认config额外添加一些默认值，得到FsDet的默认config
# adding additional default values built on top of the default values in detectron2

_CC = _C

# FREEZE Parameters
_CC.MODEL.BACKBONE.FREEZE = False # backbone默认不freeze
_CC.MODEL.PROPOSAL_GENERATOR.FREEZE = False # RPN默认不freeze
_CC.MODEL.ROI_HEADS.FREEZE_FEAT = False # RoiHead默认不freeze

# choose from "FastRCNNOutputLayers" and "CosineSimOutputLayers"
_CC.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers" # 输出层默认使用`FastRCNNOutputLayers`
# scale of cosine similarity (set to -1 for learnable scale)
_CC.MODEL.ROI_HEADS.COSINE_SCALE = 20.0 # `CosineSimOutputLayers`的cosine scale默认为20

# Backward Compatible options.
_CC.MUTE_HEADER = True
