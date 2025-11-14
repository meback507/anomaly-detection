import base64
import copy
import datetime
import io
import os
from base64 import b64encode
from random import randint
from time import sleep

import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import dash_gif_component as gif
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import scipy
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html
from joblib import dump, load
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import make_blobs, make_moons
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import IsolationForest
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline

pio.renderers.default = "svg"

pio.templates.default = "ggplot2"

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "17rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "19rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H6("Anomaly Detection", className="display-4"),
        html.Hr(),
        html.P(
            "A set of anomaly detection algorithms implemented in Python using scikit-learn",
            className="lead",
        ),
        dbc.Nav(
            [
                dbc.NavLink("Description", href="/", active="exact"),
                dbc.NavLink("Comparison", href="/page-1", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        html.P(
            "Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.",
            className="trail",
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

# Example settings
n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

# define outlier/anomaly detection methods to be compared.
anomaly_algorithms = [
    (
        "Robust covariance",
        EllipticEnvelope(contamination=outliers_fraction, random_state=36),
    ),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)),
    (
        "One-Class SVM (SGD)",
        make_pipeline(
            Nystroem(gamma=0.1, random_state=36, n_components=150),
            SGDOneClassSVM(
                nu=outliers_fraction,
                shuffle=True,
                fit_intercept=True,
                random_state=36,
                tol=1e-6,
            ),
        ),
    ),
    (
        "Isolation Forest",
        IsolationForest(contamination=outliers_fraction, random_state=42),
    ),
    (
        "Local Outlier Factor",
        LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction),
    ),
]

# Define datasets
blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
rng = np.random.RandomState(36)

dataset1 = make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5, **blobs_params)[0]
dataset2 = make_blobs(
    centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5], **blobs_params
)[0]
dataset3 = make_blobs(
    centers=[[2, 2], [-2, -2]], cluster_std=[1.5, 0.3], **blobs_params
)[0]
dataset4 = 4.0 * (
    make_moons(n_samples=n_samples, noise=0.05, random_state=0)[0]
    - np.array([0.5, 0.25])
)
dataset5 = 14.0 * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)

# Adding outliers
dataset1 = np.concatenate(
    [dataset1, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0
)
dataset2 = np.concatenate(
    [dataset2, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0
)
dataset3 = np.concatenate(
    [dataset3, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0
)
dataset4 = np.concatenate(
    [dataset4, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0
)
dataset5 = np.concatenate(
    [dataset5, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0
)


# Compare classifiers
def classifier_test(X, flag=0, theme="ggplot2"):
    plot_num = 1
    xx, yy = np.meshgrid(np.linspace(-7, 7, 50), np.linspace(-7, 7, 50))
    pio.templates.default = theme
    fig = make_subplots(
        rows=1,
        cols=5,
        subplot_titles=(
            "Robust covariance",
            "One-class SVM",
            "One-class SVM (SGD)",
            "Isolation Forest",
            "Local Outlier Factor",
        ),
    )

    for name, algorithm in anomaly_algorithms:
        algorithm.fit(X)

        # fit the data and tag outliers
        if name == "Local Outlier Factor":
            y_pred = algorithm.fit_predict(X)
        else:
            y_pred = algorithm.fit(X).predict(X)

        # plot the levels lines and the points
        if name != "Local Outlier Factor":  # LOF does not implement predict
            Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])

        y_pred[y_pred <= 0] = 0
        Z[Z <= 0] = 0

        if flag == 0:
            fig.add_trace(
                go.Scatter(
                    x=X[:, 0],
                    y=X[:, 1],
                    mode="markers",
                    marker_color=y_pred.astype(int),
                ),
                row=1,
                col=plot_num,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=xx.ravel(),
                    y=yy.ravel(),
                    mode="markers",
                    marker_color=Z.astype(int),
                ),
                row=1,
                col=plot_num,
            )
        fig.update_layout(height=425)
        fig.update_layout(showlegend=False)
        plot_num += 1

    return fig


fig1 = classifier_test(dataset1, flag=0, theme="ggplot2")
fig2 = classifier_test(dataset1, flag=1, theme="ggplot2")
fig3 = classifier_test(dataset2, flag=0, theme="presentation")
fig4 = classifier_test(dataset2, flag=1, theme="presentation")
fig5 = classifier_test(dataset3, flag=0, theme="plotly")
fig6 = classifier_test(dataset3, flag=1, theme="plotly")
fig7 = classifier_test(dataset4, flag=0, theme="ggplot2")
fig8 = classifier_test(dataset4, flag=1, theme="ggplot2")
fig9 = classifier_test(dataset5, flag=0, theme="presentation")
fig10 = classifier_test(dataset5, flag=1, theme="presentation")


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.Div(
            children=[
                dcc.Markdown(
                    """
            # **Anomaly detection algorithms**
            ##### A set set of five anomaly detection algorithms are described as well as compared here. The purpose of this webapp is to demonstrate how Plotly-Dash can be used for data visualization as well as offer a template for other interactive visualization. The five anaomaly detection algorithms are as follows:
            ### **1) Robust covariance**
            ##### Robust covariance is used to detect anomalies in datasets with Gaussian distribution. In this model, the data points away from the 3rd deviation are likely to be considered as anomalies. If every feature in a data set is Gaussian, then the statistical approach can be generalized by defining an elliptical hypersphere that covers most of the regular data points. The data points that lie away from the hypersphere can be considered as anomalies.

            ### **2) One-class Support Vector Machine (SVM)**
            ##### The basic idea of SVMs is mapping the input data points to a high-dimensional feature space and finds a hyperplane. The algorithm is chosen in such a way to maximize the distance from the closest patterns, which is called the margin. SVMs aims to minimize an upper bound of generalization error through maximizing the margin between the separating hyperplane and data. In an SVM that has one class of data points, the task is to predict a hypersphere that separates the cluster of data points from the anomalies.
            
            ### **3) One-class SVM Stochastic Gradient Descent (SGD)**
            ##### One-class SVM Stochastic Gradient Descent is an online linear version of the One-Class SVM using a stochastic gradient descent. Combined with kernel approximation techniques, it can be used to approximate the solution of a kernelized One-Class SVM, implemented in sklearn.svm.OneClassSVM, with a linear complexity in the number of samples. Note that the complexity of a kernelized One-Class SVM is at best quadratic in the number of samples. One-class SVM Stochastic Gradient Descent is thus well suited for datasets with a large number of training samples (> 10,000) for which the SGD variant can be several orders of magnitude faster.

            ### **4) Isolation Forest**
            ##### The premise of the Isolation Forest is that anomalous data points are easier to separate from the rest of the sample. In order to isolate a data point, the algorithm recursively generates partitions on the sample by randomly selecting an attribute and then randomly selecting a split value between the minimum and maximum values allowed for that attribute. Recursive partitioning can be represented by a tree structure named Isolation Tree, while the number of partitions required to isolate a point can be interpreted as the length of the path, within the tree, to reach a terminating node starting from the root.

            ### **5) Local Outlier Factor**
            ##### The local outlier factor is based on a concept of a local density, where locality is given by k nearest neighbors, whose distance is used to estimate the density. By comparing the local density of an object to the local densities of its neighbors, one can identify regions of similar density, and points that have a substantially lower density than their neighbors. These are considered to be outliers. The local density is estimated by the typical distance at which a point can be "reached" from its neighbors. The definition of "reachability distance" used in LOF is an additional measure to produce more stable results within clusters.
            """,
                    mathjax=True,
                ),
                html.Hr(),
                dcc.Markdown(
                    """
            ## **References**
            ##### 1) https://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html
            ##### 2) https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html
            ##### 3) https://scikit-learn.org/stable/modules/sgd.html#online-one-class-svm
            ##### 4) Liu, Fei Tony; Ting, Kai Ming; Zhou, Zhi-Hua (December 2008). "Isolation Forest". 2008 Eighth IEEE International Conference on Data Mining. pp. 413–422. doi:10.1109/ICDM.2008.17. ISBN 978-0-7695-3502-9. S2CID 6505449.
            ##### 5) Breunig, M. M.; Kriegel, H.-P.; Ng, R. T.; Sander, J. (2000). LOF: Identifying Density-based Local Outliers. Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data. SIGMOD.
            ##### 6) https://builtin.com/machine-learning/anomaly-detection-algorithms

            """,
                    mathjax=True,
                ),
            ]
        )
    elif pathname == "/page-1":
        return html.Div(
            children=[
                dcc.Markdown(
                    """
                # **Anomaly Detection**
                """,
                    mathjax=True,
                ),
                html.Hr(),
                dcc.Markdown(
                    """
                ### **Comparison on Dataset 1**
                #### Prediction on the points in the dataset
                """,
                    mathjax=True,
                ),
                dcc.Graph(figure=fig1),
                dcc.Markdown(
                    """
                #### Decision boundaries
                """,
                    mathjax=True,
                ),
                dcc.Graph(figure=fig2),
                html.Hr(),
                dcc.Markdown(
                    """
                ### **Comparison on Dataset 2**
                #### Prediction on the points in the dataset
                """,
                    mathjax=True,
                ),
                dcc.Graph(figure=fig3),
                dcc.Markdown(
                    """
                #### Decision boundaries
                """,
                    mathjax=True,
                ),
                dcc.Graph(figure=fig4),
                html.Hr(),
                dcc.Markdown(
                    """
                ### **Comparison on Dataset 3**
                #### Prediction on the points in the dataset
                """,
                    mathjax=True,
                ),
                dcc.Graph(figure=fig5),
                dcc.Markdown(
                    """
                #### Decision boundaries
                """,
                    mathjax=True,
                ),
                dcc.Graph(figure=fig6),
                html.Hr(),
                dcc.Markdown(
                    """
                ### **Comparison on Dataset 4**
                #### Prediction on the points in the dataset
                """,
                    mathjax=True,
                ),
                dcc.Graph(figure=fig7),
                dcc.Markdown(
                    """
                #### Decision boundaries
                """,
                    mathjax=True,
                ),
                dcc.Graph(figure=fig8),
                html.Hr(),
                dcc.Markdown(
                    """
                ### **Comparison on Dataset 5**
                #### Prediction on the points in the dataset
                """,
                    mathjax=True,
                ),
                dcc.Graph(figure=fig9),
                dcc.Markdown(
                    """
                #### Decision boundaries
                """,
                    mathjax=True,
                ),
                dcc.Graph(figure=fig10),
                html.Hr(),
                html.Hr(),
            ]
        )
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


if __name__ == "__main__":
    app.run(port=5000, debug=True, dev_tools_hot_reload=False)
