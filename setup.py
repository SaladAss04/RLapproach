from setuptools import setup, find_packages

setup(
    name="rlapproach",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'gym',
        'pytest',
        'pygame'  # Modern rendering backend
    ],
)