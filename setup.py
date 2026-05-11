from setuptools import setup, find_packages

setup(
    name="renaissance",
    packages=find_packages(
        exclude=[".dfc", ".vscode", "dataset", "notebooks", "result", "scripts", "tests"]
    ),
    package_data={"renaissance.data": ["*.json"]},
    version="1.2.0.dev0",
    license="MIT",
    description="Renaissance: A Multimodal Transformer Modeling Platform",
    keywords=["vision and language pretraining"],
    install_requires=[
        "torch",
        "torchvision",
        "transformers",
        "datasets",
        "accelerate",
        "safetensors",
        "huggingface_hub",
        "torchmetrics>=0.12",
        "omegaconf",
        "lightning",
        "pyarrow",
        "pandas",
        "einops",
        "numpy",
        "Pillow",
        "tensorboard",
    ],
)
