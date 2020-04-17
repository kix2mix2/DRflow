# DRFlow

This is a pipeline for applying Dimensionality Reduction Techniques on image/tabular datasets. It is essentially a bunch of tedious code you don't want to write yourself ;). 
 
It has support for the following tasks:
* Downloading/Uploading data from S3 
* Crawling websites using BeautifulSoup
* Resizing images
* Flattening images for DR preparation
* Data cleaning
* Applying all the DR techniques available in sklearn + UMAP
* Plotting the projections as scatterplots using images instead of dots
* Supports multi processing using Ray.

To do:
* add configuration file for the pipeline
* create automatic workflow from config
* add image augmentation techniques

## Installation Guide 

1. Create a virtual environment using the `requirements.txt` file
2. Or run `pip install  -e .` to install this package
3. Knock yourself out
4. Check the example folder to see how to use the pipeline

## Examples

#### Coil-100 dataset
![UMAP projection of COIL100](examples/coil100.png)

