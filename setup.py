import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Distribution_example_resnet",
    version="0.0.1",
    author="CDJ",
    author_email="eowns02@gmail.com",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/choidj/Distribution_example.git",
    project_urls={
        "Bug Tracker": "https://github.com/choidj/Distribution_example/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"/workspace/": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)