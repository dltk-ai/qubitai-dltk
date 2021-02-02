import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qubitai-dltk",
    version="1.0.0",
    author="DLTK",
    install_requires=[
        "pandas==1.1.3",
        "xlrd==1.2.0",
        "numpy==1.19.5",
        "requests==2.25.1",
        "openpyxl==3.0.6"
        ],
    author_email="connect@qubitai.tech",
    description="Python Client for DLTK.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dltk-ai/qubitai-dltk",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
