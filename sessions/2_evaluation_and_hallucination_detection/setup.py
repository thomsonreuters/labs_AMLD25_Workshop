from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="trlabs",
    version="0.1.0",
    description="AMLD25 Workshop Labs Package",
    author="AMLD25 Workshop Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=required,
)
