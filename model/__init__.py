# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.
from model.unet import UNet
#from model.my_unet import MyUnet
from model.unet_dilated import UNetDilated
from model.unet_with_detections import UNetWithDetections
#from model.sol_eol_finder import SOL_EOL_Finder
#from model.detector import Detector
#from model.yolo_box_detector import YoloBoxDetector
#from model.pairing_box_full import PairingBoxFull
#from model.pairing_box_net import PairingBoxNet
#from model.pairing_box_from_gt import PairingBoxFromGT
#from model.pairing_graph import PairingGraph
#from model.binary_pair_net import BinaryPairNet
#from model.binary_pair_real import BinaryPairReal
#from model.graph_net import GraphNet
#from model.line_follower import LineFollower
from model.simpleNN import SimpleNN
from model.hw_with_style import HWWithStyle
from model.rnn_test import Spacer
