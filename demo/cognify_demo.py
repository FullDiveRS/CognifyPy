#!/usr/bin/env python3
"""Thin launcher: python demo/cognify_demo.py [--all]"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cognify.demo import main

if __name__ == "__main__":
    main()
