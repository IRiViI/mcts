import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mcts",
    version="1.0.0",
    author="Rick Vink",
    author_email="rckvnk@gmail.com",
    description="MCTS, including policy value driven mcts",
    url="https://github.com/IRiViI/mcts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    keywords=["mcts", "policy", "value"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["numpy"],
)