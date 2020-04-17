from setuptools import setup, find_packages

setup(
    name="DRflow",
    version="0.9",
    packages=find_packages(),
    install_requires=[
        "pandas==0.25.3",
        "ray==0.8.0",
        "boto3==1.9.253",
        "requests==2.22.0",
        "botocore==1.12.253",
        "seaborn==0.9.0",
        "matplotlib==3.1.2",
        "numpy==1.16.2",
        "scipy==1.4.1",
        "umap_learn==0.3.10",
        "opencv_python==4.1.2.30",
        "beautifulsoup4==4.9.0",
        "Pillow==7.1.1",
        "scikit_learn==0.22.2.post1",
        "umap==0.1.1",
    ],
)