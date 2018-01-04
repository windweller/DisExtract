# -*- coding: utf-8 -*-

"""
Preprocess gigaword chinese 5th edition

We are only process text type of "story", and ignore the rest.
"story" is the most frequent type in this corpus.
"""

import os
import io
import sys
import json
import gzip
import argparse

import logging
from util import rephrase
from os.path import join as pjoin

import xml.etree.ElementTree as ET

from parser import depparse_ssplit, setup_corenlp
from cfg import SP_DISCOURSE_MARKERS
