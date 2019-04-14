from setuptools import setup



setup(
    name='aigames',
    version='0.0.1',
    description='use deep learning to play games',
    url='https://github.com/daniel-layton-wright/aigames.git',
    author='Daniel Wright',
    author_email='dlwright@alumni.stanford.edu',
    license='MIT',
    packages=[
        'aigames',
    ],
    package_data={
        'aigames': [
        ]
    },
    install_requires=[
        'torch',
        'pandas',
        'numpy',
    ],
    tests_require=[],
    setup_requires=[],
    zip_safe=True,
)