"""
Setup script for MarketLevelSC package.
"""

from setuptools import setup, find_packages

setup(
    name="MarketLevelSC",
    version="0.1.0",
    description="Market-level synthetic control methods for causal inference",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/elizabethguy86/MarketLevelSyntheticControls",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "cvxpy",
        "scikit-learn",
        "tqdm",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)