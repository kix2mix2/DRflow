import os
import cv2
import numpy as np
import pandas as pd
import ray

@ray.remote
def flatten_images(input_folder, output_folder, filename):
    # check if input folder exists
    if not os.path.exists(input_folder):
        print("Input folder not found: {}".format(input_folder))
        return 1

    # check if output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # check if flat file exists, skip processing it if it does
    if os.path.exists(output_folder + filename):
        print("Skipping file: {}".format(filename))
        return

    images = []
    names = []

    for i, ff in enumerate(os.listdir(input_folder)):
        if ff.startswith("."):
            # print('not an image: ' + ff)
            continue
        # print(i, ff.split(".")[0])

        img = cv2.imread(input_folder + ff, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("This image is None: " + ff)
            continue

        if len(img.shape) > 2 and img.shape[2] == 4:
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            except:
                continue

        img = pd.DataFrame(np.array(img).flatten()).T
        # print(img)
        images.append(img)
        names.append(ff.split(".")[0])

    if len(images) > 0:
        imgs = pd.concat(images, axis=0)
        imgs["filename"] = names
        imgs["labels"] = ["_".join(n.split("_")[:-1]) for n in names]
        imgs = imgs.dropna(axis=1)
        # print(imgs.shape)
        # print(output_folder + filename + ".csv")
        imgs.to_csv(output_folder + filename + ".csv", index=False)
        # print(imgs.head(2))

    print("----Finished: {}----".format(input_folder))

    return 0


def flatten_image(data):
    d = data.shape
    # print(d)
    flat_data = data.flatten().reshape(d[0], np.product(d[1:]))
    # print(flat_data.shape)
    return flat_data

@ray.remote
def keep_top_k_classes(filename, c=10):
    output_name = "{}_{}classes.csv".format(filename.split(".csv")[0], c)

    if os.path.exists(output_name):
        print("Skipping: {}".format(output_name))
        return

    try:
        df = pd.read_csv(filename)
        print("Started: {}".format(filename))
    except:

        # print("{} didnt work".format(filename))
        return



    print("Before: {}".format(df.shape))
    df_agg = (
        df.groupby(["labels"])
        .count()["filename"]
        .reset_index()
        .sort_values(["filename"], ascending=False)
    )


    for i in range(2,c+2,2):
        ll = list(df_agg.reset_index(drop=True).loc[:i-1, "labels"])
        # print('aasdasdadasdadsasd')
        print(i, ll)
        # print("{} \n ---- \n {}".format(filename, df_agg.head(i)))
        df1 = df.loc[df["labels"].isin(ll)].copy()

        output_name = "{}_{}classes.csv".format(filename.split(".csv")[0], i)


        df1.to_csv(output_name, index=False)
        print("After: {}".format(df1.shape))
        print("Finished: {}".format(filename))
    return 1
