#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for jk_fkik

灵巧手拇指和四指正逆运动学求解库
适用于树莓派 5 等嵌入式平台
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="jk_fkik",
    version="1.0.0",
    author="上海钧舵",
    description="灵巧手拇指和四指正逆运动学求解库",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Proprietary",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "scipy>=1.7.0",
            "pytest>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "jk-fkik-examples=examples:main",
        ],
    },
)
