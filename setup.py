#!/usr/bin/env python3
"""
Multi-Sensor SNN-Fusion Setup Configuration
Neuromorphic computing framework for real-time multi-modal sensor fusion
"""

from setuptools import setup, find_packages

# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="snn-fusion",
    version="0.1.0",
    author="Terragon Labs",
    author_email="research@terragonlabs.com",
    description="Neuromorphic multi-modal sensor fusion framework with spiking neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/terragonlabs/Multi-Sensor-SNN-Fusion",
    project_urls={
        "Bug Tracker": "https://github.com/terragonlabs/Multi-Sensor-SNN-Fusion/issues",
        "Documentation": "https://snn-fusion.readthedocs.io/",
        "Research Paper": "https://arxiv.org/abs/2025.xxxxx",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
        "hardware": [
            "lava-dl>=0.5.0",
            "akida>=2.0.0",
            "pyNN>=0.10.0",
        ],
        "visualization": [
            "matplotlib>=3.6.0",
            "seaborn>=0.12.0",
            "plotly>=5.0.0",
            "dash>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "snn-fusion=snn_fusion.cli:main",
            "snn-train=snn_fusion.training.cli:main",
            "snn-deploy=snn_fusion.hardware.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "snn_fusion": [
            "models/pretrained/*.pt",
            "datasets/configs/*.yaml",
            "hardware/configs/*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "neuromorphic",
        "spiking neural networks",
        "sensor fusion",
        "liquid state machines",
        "multi-modal",
        "real-time",
        "low-latency",
        "edge computing",
        "robotics",
        "autonomous systems",
    ],
)