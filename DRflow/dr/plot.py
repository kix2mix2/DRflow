import matplotlib.pyplot as plt
import pandas as pd
# import modin.pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
import ray


# ray.init()
def plot_all_dr_files(
        input_folder, output_folder, thumbnail_folder, size_limit=1000, color_palette="Set2"
):
    if not os.path.exists(input_folder):
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    files = os.listdir(input_folder)

    color_palette = sns.color_palette(color_palette)
    print(len(color_palette))

    for i, file in enumerate(files):
        print(file)
        if not file.endswith(".csv"):
            continue
        try:
            if i == 0:
                df = pd.read_csv(input_folder + file)
                print(df.shape)

                if df.shape[0] > size_limit:
                    _, df = train_test_split(
                        df, test_size = size_limit, stratify = df["labels"]
                    )

                le = preprocessing.LabelEncoder()

                labels = le.fit_transform(df["labels"])
                paths = list(thumbnail_folder + df["filename"] + ".png")
                idx = df.index

            plot_projection(
                input_folder + file, output_folder, idx, labels, paths, color_palette
            )

        except:
            print("fail")
            continue


def getImage(path, cmap=None):
    if cmap is None:
        return OffsetImage(plt.imread(path))
    else:
        return OffsetImage(plt.imread(path), cmap = cmap)


@ray.remote
def plot_projection(input_file, output_folder, thumbnail_folder, color_palette=None, cmap=None, loc=[0, 1],
                    fs=(100, 100), size_limit=500):
    print(input_file)
    output_file = output_folder + input_file.split("/")[-1].split(".csv")[0] + ".png"
    # if os.path.exists(output_file):
    #     # print('skipping')
    #     return
    try:
        df = pd.read_csv(input_file)
    except:
        print("file not found")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if df.shape[0] > size_limit:
        _, df = train_test_split(
            df, test_size = size_limit, stratify = df["labels"]
        )

    df["paths"] = thumbnail_folder + df['filename'] + '.png'
    le = preprocessing.LabelEncoder()
    df["ll"] = le.fit_transform(df['labels'])
    # print(df.head())
    fig, ax = plt.subplots(figsize = fs)
    ax.scatter(df.iloc[:, loc[0]], df.iloc[:, loc[1]])

    for i, row in df.iterrows():
        try:
            ab = AnnotationBbox(getImage(row["paths"], cmap), (row[loc[0]], row[loc[1]]), frameon = True)
        except Exception as e:
            # print(e)
            # print('failed1', row["paths"])
            try:
                ab = AnnotationBbox(
                    getImage(row["paths"].split(".png")[0] + ".jpg"),
                    (row[loc[0]], row[loc[1]]),
                    frameon = True,
                )
            except:
                #                 print('failed2')
                continue

        ab.patch.set_linewidth(3)
        ab.patch.set_edgecolor(#'white',
                               color_palette[
                                   int(row["ll"])
                                   if row["ll"] < len(color_palette)
                                   else len(color_palette) - 1
                               ]
                               )
        ab.patch.set_facecolor(#'white',
                               color_palette[
                                   int(row["ll"])
                                   if row["ll"] < len(color_palette)
                                   else len(color_palette) - 1
                               ]
                               )
        ax.add_artist(ab)

    print('--------', output_file)
    # plt.show()
    plt.savefig(output_file)
    plt.close()
