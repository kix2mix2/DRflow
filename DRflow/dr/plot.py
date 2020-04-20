import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns


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
                        df, test_size=size_limit, stratify=df["labels"]
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


def getImage(path):
    return OffsetImage(plt.imread(path))


def plot_projection(input_file, output_folder, idx, labels, paths, color_palette, loc=[0,1], fs= (100, 100)):
    print(input_file)
    try:
        df = pd.read_csv(input_file)
    except:
        print("file not found")
        return

    df = df.loc[idx, :]
    df["paths"] = paths
    df["ll"] = labels
    fig, ax = plt.subplots(figsize=fs)
    ax.scatter(df.loc[:, loc[0]], df.loc[:, loc[1]])

    for i, row in df.iterrows():
        try:
            ab = AnnotationBbox(getImage(row["paths"]), (row[loc[0]], row[loc[1]]), frameon=True)
        except:
            try:
                ab = AnnotationBbox(
                    getImage(row["paths"].split(".png")[0] + ".jpg"),
                    (row[loc[0]], row[loc[1]]),
                    frameon=True,
                )
            except:
                continue

        ab.patch.set_linewidth(.25)
        ab.patch.set_edgecolor( 'white'
            # color_palette[
            #     int(row["ll"])
            #     if row["ll"] < len(color_palette)
            #     else len(color_palette) - 1
            # ]
        )
        ab.patch.set_facecolor('white'
            # color_palette[
            #     int(row["ll"])
            #     if row["ll"] < len(color_palette)
            #     else len(color_palette) - 1
            # ]
        )
        ax.add_artist(ab)

    output_file = output_folder + input_file.split("/")[-1].split(".csv")[0] + ".pdf"
    print('--------')
    plt.show()
    plt.savefig(output_file)
    plt.close()
