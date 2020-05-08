import os

import pandas as pd
import ray
import umap

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, MDS, TSNE
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.manifold import SpectralEmbedding as SE
from sklearn.random_projection import GaussianRandomProjection as GRP
from sklearn.model_selection import train_test_split


# @ray.remote
def apply_dr(
    input_file,
    output_folder,
    dataset_name="MNIST",
    dr_name="PCA",
    perplexity=None,
    n_neighbors=None,
    min_dist=None,
    max_samples=10000, size = None, c = None
):
    fn = "{dataset_name}{size}{c}{dr_name}{perp}{neigh}{mindist}".format(
        dataset_name=dataset_name,
        size = "_size" +str(size) if size is not None else "",
        c = "_c" +str(c) if c is not None else "",
        dr_name="_" + dr_name,
        perp="_p" + str(perplexity) if perplexity is not None else "",
        neigh="_n" + str(n_neighbors) if n_neighbors is not None else "",
        mindist="_d" + str(min_dist) if min_dist is not None else "",
    )

    if os.path.exists(output_folder + fn + ".csv"):
        print("---------Skipping: {}{}-----------".format(input_file, fn))
        return


    try:
        df = pd.read_csv(input_file)
        print(("---------Startings: {} - {}-----------".format(input_file, fn)))
    except:
        print("{} - does not exist".format(fn))
        return

    y = df["labels"]
    X = df.iloc[:, :-2]

    if df.shape[0] > max_samples:
        X_train, features, y_train, labels = train_test_split(
            X, y, test_size=max_samples, random_state=42, stratify=y
        )
    else:
        features = X
        labels = y

    idx = list(features.index)
    filename = df.loc[idx, "filename"]
    ########

    ## apply dr
    if dr_name == "CPCA":
        dr = CPCA(n_components=2)

    if dr_name == "PCA":
        dr = PCA(n_components=2)

    elif dr_name == "TSNE":
        dr = TSNE(n_components=2, perplexity=perplexity, verbose=0)

    elif dr_name == "ISM":
        dr = Isomap(n_components=2, n_neighbors=n_neighbors)

    elif dr_name == "LLE":
        dr = LLE(n_components=2, n_neighbors=n_neighbors)

    elif dr_name == "SE":
        dr = SE(n_components=2, n_neighbors=n_neighbors)

    elif dr_name == "UMAP":
        dr = umap.UMAP(
            n_components=2, n_neighbors=n_neighbors, verbose=False, min_dist=min_dist
        )

    elif dr_name == "GRP":
        dr = GRP(n_components=2)

    elif dr_name == "MDS":
        dr = MDS(n_components=2)

    try:
        dr_data = dr.fit_transform(features)
    except:
        return

    dr_data = pd.DataFrame(
        dr_data, columns=["{}_1".format(dr_name), "{}_2".format(dr_name)]
    )
    dr_data.index = idx

    ## save stuff
    if labels is not None:
        dr_data["labels"] = list(labels)
        dr_data["filename"] = list(filename)

        # fig, ax = plt.subplots()
        # sns.scatterplot(dr_data['{}_1'.format(dr_name)], dr_data['{}_2'.format(dr_name)], hue = dr_data['labels'], ax=ax)
        # plt.savefig(dataset_name + '/figures/1_' + fn +'.pdf')
        # plt.close('all')

    dr_data.to_csv(output_folder + fn + ".csv", index=False)
    print(("---------Finished: {}{}-----------".format(dataset_name, fn)))

    return


def load_and_combine(folder):
    files = os.listdir(folder)
    datasets = []
    for f in files:
        datasets.append(pd.read_csv(folder + files))

    return pd.concat(datasets, axis=1)
