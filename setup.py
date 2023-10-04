from setuptools import find_packages, setup

setup(
    name='ylib',
    version='0.0.1',
    description='zhangsihao yang lib for python',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
