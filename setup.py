from setuptools import setup, find_packages

setup(
    name="DRflow",
    version="0.9",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "ray",
        "boto3",
        "requests",
        "botocore",
        "seaborn",
        "matplotlib",
         "numpy",
        "scipy",
        "umap_learn",
        "opencv_python",
        "beautifulsoup4",
        "Pillow",
        "scikit_learn",
        "umap",
    ],
)
