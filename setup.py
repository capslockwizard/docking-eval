import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="docking-eval",
    version="0.7",
    author="Justin Chan",
    author_email="capslockwizard@gmail.com",
    description="Evaluates docking poses based on the CAPRI criteria",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/capslockwizard/docking-eval",
    packages=setuptools.find_packages(),
    install_requires=['numba', 'numpy', 'pandas', 'u-msgpack-python', 'MDAnalysis', 'drsip-common'],
    classifiers=[
        "Environment :: Plugins",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
