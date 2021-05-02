from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="pyai-cs",
    version="0.0.4",
    author="Claudio Scheer",
    author_email="claudioscheer@protonmail.com",
    description="My own artificial intelligence library intended for learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lab-a1/pyai",
    project_urls={
        "Bug Tracker": "https://github.com/lab-a1/pyai/issues",
    },
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Public Domain",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    python_requires=">=3.6",
    packages=find_packages(where="src", exclude=["tests*"]),
)
