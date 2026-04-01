from setuptools import setup, find_packages

setup(
    name="kyb-graph-analytics",
    version="0.1.0",
    description="Graph-based fraud detection for KYB/AML: shell company and hidden ownership detection",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.8",
    install_requires=[
        "networkx>=3.0",
        "python-louvain>=0.16",
        "numpy>=1.24",
        "scipy>=1.10",
    ],
)
