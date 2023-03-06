import os
from dflow import config

__author__ = 'Jiahao Fan'
__author_email__ = 'jiahaofan@pku.edu.cn'
try:
    from ._version import version as __version__
except ImportError:
    from .__about__ import __version__

SRC_ROOT = __path__[0]
config["extender_image_pull_policy"] = "IfNotPresent"
config["util_image_pull_policy"] = "IfNotPresent"
config["dispatcher_image_pull_policy"] = "IfNotPresent"