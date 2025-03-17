#/usr/bin/env python3
# -*- coding: utf-8 -*-

import seaborn as sns
import numpy as np
import matplotlib.pylab as plt
from utils import simulate_dataset, format_dataset
from repairs import DI_list_geometric_repair, DI_list_random_repair
from sklearn.manifold import TSNE
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def plot(X0, X1, clf):
    # Create a dataframe
    data = []
    for trial in tqdm(range(30)):
        res_geometric = DI_list_geometric_repair(X0, X1, clf=clf)
        res_random = DI_list_random_repair(X0, X1, clf=clf)
        for i, (geo, rand) in enumerate(zip(res_geometric, res_random)):
            data.append(['Geometric', trial, geo, i / (len(res_geometric) - 1)])
            data.append(['Random', trial, rand, i / (len(res_random) - 1)])

    df = pd.DataFrame(data, columns=['Repair', 'Trial', 'DI', 'Lambda'])

    # Plot using seaborn
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=df, x='Lambda', y='DI', hue='Repair')
    plt.title('DI as a function of lambda')
    plt.tight_layout()
    plt.savefig(f'DI_lambda_{clf}.png')
    plt.show()

if __name__ == '__main__':
    # Parameters for the simulation
    n0 = 600
    n1 = 400
    mu0 = (3, 3, 2, 2.5, 3.5)
    mu1 = (4, 4, 3, 3.5, 4.5)
    sigma = np.diag([1, 1, 0.5, 0.5, 1])
    beta0 = (1, -1, -0.5, 1, -1)
    beta1 = (1, -0.4, 1, -1, 1)

    # Simulate the dataset
    X0, X1, Y0, Y1 = simulate_dataset(n0, n1, mu0, mu1, sigma, beta0, beta1)
    X0.shape, X1.shape, Y0.shape, Y1.shape

    X,Y = format_dataset(X0, X1, Y0, Y1)

    X_embedded = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(X[:,1:])
    X_embedded = np.concatenate([X[:,:1], X_embedded], axis=1)
    print(X_embedded.shape)

    X,Y = format_dataset(X0, X1, Y0, Y1)
    print(X.shape, Y.shape)
    plt.figure(1, (5, 5))
    plt.plot(X_embedded[X_embedded[:,0] == 0][:, 1], X_embedded[X_embedded[:,0] == 0][:, 2], '+')
    plt.plot(X_embedded[X_embedded[:,0] == 1][:, 1], X_embedded[X_embedded[:,0] == 1][:, 2], 'x')
    plt.title('Embedding of the dataset')
    plt.savefig('embedding.png')
    # plt.show()

    plot(X0, X1, clf=LogisticRegression(random_state=69))
    plot(X0, X1, clf=RandomForestClassifier(random_state=69, n_estimators=10, max_depth=5))

