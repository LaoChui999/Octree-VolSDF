import torch
import numpy as np
import math
from torch import nn, autograd
from svox.helpers import N3TreeView, DataFormat, LocalIndex, _get_c_extension

class LONodeA(N3TreeView):
    def __init__(self, tree, key):
        super(LONodeA, self).__init__(tree, key)

        self.Position=key

    def RefineCorners(self):
        ret = self.tree.RefineCorners(1, sel=self._unique_node_key())
        return ret