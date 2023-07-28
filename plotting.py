import seaborn as sns
import matplotlib.pyplot as plt

def plot_1000_comp(df):
    fig, ax = plt.subplots()
    sns.scatterplot(df, y='answer', x='predict', ax=ax, alpha=0.1, s=3)

    max_limit = 1000
    ax.set_xlim(0, max_limit)
    ax.set_ylim(0, max_limit)
    plt.plot(range(max_limit), range(max_limit), 'firebrick')
    return fig

def plot_6500_comp(df):
    fig, ax = plt.subplots()
    sns.scatterplot(df, y='answer', x='predict', ax=ax, alpha=0.1, s=1)

    max_limit = 6500
    ax.set_xlim(0, max_limit)
    ax.set_ylim(0, max_limit)
    plt.plot(range(max_limit), range(max_limit), 'firebrick')
    return fig