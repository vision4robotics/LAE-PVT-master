# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
#from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from pysot.tracker.siamrpn_tracker_f import SiamRPNTracker
from pysot.tracker.siammask_tracker_f import SiamMaskTracker
from pysot.tracker.siamrpnlt_tracker import SiamRPNLTTracker
from pysot.tracker.dsiamrpn import DSiamRPNTracker

TRACKS = {
          'SiamRPNTracker': SiamRPNTracker,
          'SiamMaskTracker': SiamMaskTracker,
          'SiamRPNLTTracker': SiamRPNLTTracker,
          'DSiamRPNTracker': DSiamRPNTracker
         }


def build_tracker_f(model):
    return TRACKS[cfg.TRACK.TYPE](model)
