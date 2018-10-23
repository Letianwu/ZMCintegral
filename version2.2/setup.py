from setuptools import setup

setup(
    name="ZMCintegral",
    packages=['ZMCintegral'],
    version="2.2",
    description="Easy to use python package for Multidimensional-multi-gpu Monte Carlo integration",
    author='ZHANG Junjie',
    author_email='zjacob@mail.ustc.edu.cn',
    url='https://github.com/Letianwu/ZMCintegral',
    keywords=['Monte Carlo integraion','multi-gpu'],
    install_requires=['tensorflow-gpu'],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: Apache2.0',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Linux',
        'Programming Language :: Python :: 3.3'
    ]
)
