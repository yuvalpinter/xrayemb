from setuptools import setup, find_packages

setup(
    name='xrayemb',
    version='0.0.1',
    url='https://github.com/yuvalpinter/xrayemb',
    description='XRayEmb implementation in distributed pytorch',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "transformers==2.8.0",
        "numpy==1.16.4",
        "torch==1.8.0",
        "sentencepiece==0.1.91",
        "tensorboard==1.15",
        "scikit-learn",
        "pandas",
        "tqdm"
    ],
)
