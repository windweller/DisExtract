"""
We generate why corpus based on
the test corpus of L2E
"""

import os
import json
import argparse
from os.path import join as pjoin
from os.path import dirname, abspath

import numpy as np
import random

parser = argparse.ArgumentParser(description='Generate Why Pairs')

args, _ = parser.parse_known_args()
