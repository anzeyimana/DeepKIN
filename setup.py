from setuptools import setup, find_packages

VERSION = '0.1.0'
DESCRIPTION = 'DeepKIN Toolkit'
LONG_DESCRIPTION = 'A deep learning toolkit for Kinyarwanda NLP.'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="deepkin",
    version=VERSION,
    author="Antoine Nzeyimana",
    author_email="<nzeyi@kinlp.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        "cffi",
        "progressbar2",
        "seqeval",
        "youtokentome",
        "tensorboardX",
        "Cython",
        "distro",
        "sacremoses",
        "fastBPE",
        "packaging",
        "mutagen",
        "torchmetrics",
    ],
    keywords=['python', 'deepkin', 'kinyarwanda', 'nlp', 'deep learning'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Research and Development",
        "Programming Language :: Python :: 3",
        "Operating System :: Linux :: Linux OS",
    ]
)
