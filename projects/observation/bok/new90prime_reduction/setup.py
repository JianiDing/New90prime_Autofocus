from setuptools import setup, find_packages

setup(
    name="ccdtools",
    version="0.0.1",
    description="CCD utilities and bad-pixel masking tools",
    packages=find_packages(where='.'),
    include_package_data=True,
    install_requires=[
        "numpy>=1.21",
        "astropy",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "ccdtools=ccdtools.cli:main",
        ],
    },
)
