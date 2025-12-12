import os
from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name="IC2",
    version="0.0.1",
    author="Jinling Yan",
    author_email="haust_yjl@163.com",
    url='https://github.com/JinlingY/IC2',
    description="Interventional Dynamical Causality under Latent Confounders",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.11',
    install_requires=[
        'numpy',
        'torch',
        'matplotlib',
        'scikit-learn',
        'seaborn',
        'pandas',
        'networkx',
        'scipy',
        'numba'
    ],
    license="MIT Licence",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ]
)
