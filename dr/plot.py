import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


# ray.init()


def getImage(path):
    return OffsetImage(plt.imread(path))


def plot_projection(input_file, output_folder, idx,  labels, paths, color_palette):
    print(input_file)
    try:
        df = pd.read_csv(input_file)
    except:
        print('file not found')
        return

    df = df.loc[idx, :]
    df['paths'] = paths
    df['ll'] = labels
    fig, ax = plt.subplots(figsize = (100, 100))
    ax.scatter(df.iloc[:, 0], df.iloc[:, 1])

    for i, row in df.iterrows():
        try:
            ab = AnnotationBbox(getImage(row['paths']), (row[0], row[1]), frameon = True)
        except:
            try:
                ab = AnnotationBbox(getImage(row['paths'].split('.png')[0] + '.jpg'), (row[0], row[1]), frameon = True)
            except:
                continue

        ab.patch.set_linewidth(4)
        ab.patch.set_edgecolor(color_palette[int(row['ll']) if row['ll'] < len(color_palette) else len(color_palette)-1])
        ab.patch.set_facecolor(color_palette[int(row['ll']) if row['ll'] < len(color_palette) else len(color_palette)-1])
        ax.add_artist(ab)

    output_file = output_folder + input_file.split('/')[-1].split('.csv')[0] + '.pdf'
    print(output_file)
    plt.savefig(output_file)
    plt.close()
