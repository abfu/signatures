from setuptools import setup
import setuptools

setuptools.setup(
    name='sig',
    version='0.1',
    install_requires=['torch',
                      'torchvision',
                      'pandas',
                      'numpy',
                      'notebook',
                      'scikit-image',
                      'pytest']
)
