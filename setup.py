from setuptools import setup, find_packages

setup(
    name="renaissance",
    packages=find_packages(
        exclude=[".dfc", ".vscode", "dataset", "notebooks", "result", "scripts"]
    ),
    version="1.1.0",
    license="MIT",
    description="Renaissance: A Multimodal Transformer Modeling Platform",
    keywords=["vision and language pretraining"],
    install_requires=["torch", "lightning"],
)
