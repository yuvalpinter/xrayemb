from setuptools import setup, find_packages

setup(
    name='tokdetok_release',
    version='0.0.1',
    url='https://bbgithub.dev.bloomberg.com/ypinter/tokdetok_release',
    description='TokDetok implementation in distributed pytorch',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "transformers==2.8.0",
        "numpy==1.16.4",
        "torch==1.6.0",
        "sentencepiece==0.1.91",
        "tensorflow",
        "tensorboard==1.15",
        "scikit-learn",
        "pandas",
        "tqdm"
    ],
)
