import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sparsemod", # Replace with your own username
    version="0.1.2",
    author="Juan Borrego Carazo",
    author_email="juan.borrego@uab.cat",
    description="Package for automatically creating performing but low sized neural networks suitable for microcontrollers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BCJuan/SpArSeMod",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
