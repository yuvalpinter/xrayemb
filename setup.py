from setuptools import setup, find_packages

setup(
    name='xrayemb',
    version='0.0.1',
    url='https://github.com/yuvalpinter/xrayemb',
    description='XRayEmb implementation in distributed pytorch',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "transformers==4.19.2",
        "numpy==1.21.6",
        "torch==1.10.1",
        "sentencepiece==0.1.91",
        "tensorboard==1.15",
        "scikit-learn",
        "pandas",
        "tqdm"
    ],
)
