from setuptools import setup, find_packages


setup(
    name='aigames',
    version='0.0.1',
    description='use deep learning to play games',
    url='https://github.com/daniel-layton-wright/aigames.git',
    author='Daniel Wright',
    author_email='dlwright@alumni.stanford.edu',
    license='MIT',
    packages=find_packages(exclude=['tests']),
    package_data={
        'aigames': [
            'experiments/alpha/hearts/config/*',
        ]
    },
    install_requires=[
        'torch', 'torchvision',
        'pandas',
        'numpy',
        'google-cloud-storage',
        'tensorboardX', 'matplotlib', 'optuna', 'wandb', 'pytorch_lightning', 'gcsfs', 'tqdm', 'json-fix'
    ],
    tests_require=[],
    setup_requires=[],
    zip_safe=True,
)
