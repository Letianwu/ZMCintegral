import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ZMCintegral",
    version="5.1",
    author="Cao Xiao-Yan, ZHANG Jun-Jie",
    author_email="zjacob@mail.ustc.edu.cn",
    description="An easy way to use multi-GPUs to calculate multi-dimensional integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Letianwu/ZMCintegral",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.7',
)
