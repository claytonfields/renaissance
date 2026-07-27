from pathlib import Path

from setuptools import find_packages, setup

# Single-source the dependency list from requirements.txt so the two
# never drift. Skips comments, blank lines, and -r/-e directives.
_REQ_FILE = Path(__file__).parent / "requirements.txt"
install_requires = [
    line
    for raw in _REQ_FILE.read_text().splitlines()
    if (line := raw.strip()) and not line.startswith(("#", "-"))
]

setup(
    name="renaissance",
    packages=find_packages(
        exclude=[".dfc", ".vscode", "dataset", "notebooks", "result", "scripts", "tests", "v2", "v2.*"]
    ),
    package_data={"renaissance.data": ["*.json"]},
    version="1.3.0.dev0",
    license="MIT",
    description="Renaissance: A Multimodal Transformer Modeling Platform",
    keywords=["vision and language pretraining"],
    install_requires=install_requires,
)
