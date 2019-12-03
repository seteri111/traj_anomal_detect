import pandas as pd
from matplotlib import pyplot as plt

from line_seg import get_AIS_data

def draw_traj(ax, df):
    for idx in df.id.drop_duplicates():
        ax.plot(df.loc[df.id == idx, 'LON'], df.loc[df.id == idx, 'LAT'], 'g-', lw=0.5, alpha=0.2)

if __name__ == "__main__":
    df = get_AIS_data()
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    draw_traj(ax, df)
    plt.show()