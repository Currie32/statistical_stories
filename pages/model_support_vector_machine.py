from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from dash import callback, dcc, html, Input, Output, register_page
from plotly.graph_objs._figure import Figure as plotly_figure
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


register_page(__name__, path="/support-vector-machine")

layout = html.Div(className='content', children=[
    html.H1(className='content-title', children='Support Vector Machine (SVM)'),
    html.Div(
        className="resource-link",
        children=[html.A("Link to scikit-learn", target="_blank", href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html")]
    ),
    html.H2(className='section-title', children='Overview'),
    html.Div(className='paragraph', children=[
        html.P("Once upon a time, there was a sports fanatic named Dave. Dave was passionate about soccer, and he decided to organize a friendly match at the local field. Little did he know that this soccer match would become the perfect setting for him to understand the Support Vector Machine (SVM) algorithm."),
        html.P("Dave divided the players into two teams, Team Red and Team Blue, and positioned them on opposite sides of the field. His goal was to find the best way to separate the players so that there was a clear boundary between the two teams."),
        html.P("In the world of SVM, this boundary is called the hyperplane. Dave envisioned it as an invisible line right in the middle of the field. The players were scattered around, so the challenge was to draw this line in such a way that it maximally separates the players from both teams."),
        html.P("As Dave thought about this, he realized that the shape of the hyperplane could be crucial. He wanted it to be equidistant from players of both teams to create a fair and effective separation. This concept is known as maximizing the margin, and it's a fundamental idea behind SVM."),
        html.P("Dave quickly noticed that some players were positioned closer to the center, while others were closer to the sidelines. To ensure the best separation, he identified one player from each team who was closest to the hyperplane. These players were called support vectors because they played a crucial role in determining the optimal position of the hyperplane."),
        html.P("Now, Dave faced an interesting challenge. Once the game started and the players began moving around the field, the teams were not always neatly separated by a straight line. So, he needed a way to deal with non-linear boundaries. Enter the kernels!"),
        html.P("Dave discovered three types of kernels – linear, polynomial, and radial basis function (RBF). The linear kernel worked well when the players were roughly separated by a straight line. The polynomial kernel allowed for more flexibility in handling curved separations, and the RBF kernel was excellent for complex, irregular patterns on the field."),
        html.P("Throughout the game, Dave could often successfully separate the teams using the SVM algorithm with the appropriate kernel. This led to a great day for everyone. The players enjoyed their game of soccer and Dave was pleased to have learned about the SVM algorithm."),
    ]),
    html.H2(className='section-title', children='Summary'),
    html.Div(className='paragraph', children=[
        html.P("The Support Vector Machine (SVM) algorithm works by finding the optimal hyperplane that maximally separates different classes in a dataset. It achieves this by using support vectors as key data points for defining the decision boundary.")
    ]),
    html.H2(className='section-title', children='Visualizing the SVM algorithm'),
    html.Div(className='paragraph-italic', children=[
        html.P(children=["The algorithm's parameters are defined below the plot."]),
    ]),
    html.Button('Generate New Data', id='button-new-data', n_clicks=0),
    html.Div(className='plot-parameters', children=[
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='C'),
            dcc.Input(className='parameter-value', id='input-c-svm', value=1, min=0.1, max=100, step=0.1, type='number'),
            html.Label(className='parameter-label', children='gamma'),
            dcc.Dropdown(
                id='input-gamma-svm',
                value='scale',
                options=[
                    {'label': 'auto', 'value': 'auto'},
                    {'label': 'scale', 'value': 'scale'},
                ],
                clearable=False,
            ),
        ]),
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='kernel'),
            dcc.Dropdown(
                id='input-kernel-svm',
                value='rbf',
                options=[
                    {'label': 'linear', 'value': 'linear'},
                    {'label': 'poly', 'value': 'poly'},
                    {'label': 'rbf', 'value': 'rbf'},
                    {'label': 'sigmoid', 'value': 'sigmoid'},
                ],
                clearable=False,
            ),
            html.Label(className='parameter-label', children='coef0'),
            dcc.Input(className='parameter-value', id='input-coef0-svm', value=0, min=-100, max=100, step=0.1, type='number'),
        ]),
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='degree'),
            dcc.Input(className='parameter-value', id='input-degree-svm', value=3, min=1, max=100, step=1, type='number'),
        ]),
    ]),
    html.Div(className='plot-full-width', children=[dcc.Graph(id='plot-svm')]),
    html.H2(className='section-title', children='Parameters'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("C: "), "A regularization parameter that controls the trade-off between a smooth decision boundary and accurate classification, with higher values emphasizing correct classification, which can lead to a more complex decision boundary."]),
        html.P(children=[html.Strong("kernel: "), "The type of kernel to use, influencing the shape of the decision boundary."]),
        html.P(children=[html.Strong("degree: "), "Only for the polynomial kernel, it determines the degree of the polynomial, impacting the complexity of the relationship captured."]),
        html.P(children=[html.Strong("gamma: "), "Defines the reach of a single training example's influence, affecting the shape of the decision boundary; higher values lead to a more intricate boundary."]),
        html.P(children=[html.Strong("coef0: "), "Only for the polynomial and sigmoid kernels, it adjusts the impact of higher-order terms in the kernel function."]),
    ]),
    html.H2(className='section-title', children='Assumptions'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("1. Feature Independence: "), "Features are independent of each other. If features are highly correlated, it might affect the algorithm's performance."]),
        html.P(children=[html.Strong("2. Balanced Classes: "), "Performance tends to be better when the classes are balanced, meaning that there is roughly an equal number of instances for each class. Imbalanced datasets might lead to biased models, and additional techniques like class weighting or resampling may be needed."]),
        html.P(children=[html.Strong("3. Outliers Have Limited Impact: "), "The algorithm aims to maximize the margin, so depending on the values of the outlier, it can potentially impact the decision boundary."]),
        html.P(children=[html.Strong("4: Homoscedasticity: "), "The variance of the features is similar across all classes. In cases where the variance significantly differs between classes, the algorithm might be influenced more by the class with higher variance."]),
    ]),
    html.H2(className='section-title', children='Appropriate use cases'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("Classification: "), "Suitable for both binary and multiclass classification."]),
        html.P(children=[html.Strong("High-Dimensional Data: "), "Effective in high-dimensional spaces and is relatively memory efficient. This includes when the number of features is greater than the number of samples. Optimizing the C parameter helps avoid overfitting."]),
        html.P(children=[html.Strong("Non-Linear Relationships: "), "When using a non-linear kernel, SVMs are suitable for capturing complex, non-linear relationships in the data."]),
        html.P(children=[html.Strong("Anomaly Detection: "), "Effective in identifying anomalies in data, making them suitable for applications such as fraud detection or network intrusion detection."]),
    ]),
    html.H2(className='section-title', children='Inappropriate use cases'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("Large Datasets: "), "Training an SVM can become computationally expensive  and not scale well as the data significantly increases in size."]),
        html.P(children=[html.Strong("Interpretability: "), "SVMs can be complex and challenging to interpret, especially when using a non-linear kernel."]),
        html.P(children=[html.Strong("Data with High Noise: "), "Does not perform well when the dataset has more noise, i.e., when target classes are overlapping."]),
    ]),
])


@callback(
    Output('plot-svm', 'figure'),
    Input('input-c-svm', 'value'),
    Input('input-kernel-svm', 'value'),
    Input('input-degree-svm', 'value'),
    Input('input-gamma-svm', 'value'),
    Input('input-coef0-svm', 'value'),
    Input('button-new-data', 'n_clicks'),
)
def k_nearest_neighbors(
    c: float,
    kernel: str,
    degree: int,
    gamma: str,
    coef0: float,
    random_state: int,
) -> Tuple[plotly_figure, plotly_figure]:
    """
    Generate training and testing data for an SVM model.
    Train the model and make predictions.
    Plot the results.
    """

    # Step size when plotting the prediciton boundary
    step = 0.1

    # Generate a synthetic dataset
    x, y = make_classification(
        n_samples=200, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=2, random_state=random_state,
    )

    # Split the dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=random_state)

    # Fit the SVM classifier on the training data
    svm = SVC(
        C=c,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        coef0=coef0,
        random_state=random_state,
    )
    svm.fit(x_train, y_train)
    preds = svm.predict(x_test)

    # Generate a meshgrid to visualize the decision boundary
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

    # Predict the class labels for each point in the meshgrid
    preds_mesh = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    preds_mesh = preds_mesh.reshape(xx.shape)

    # Create a scatter plot of the training data
    scatter_train = go.Scatter(
        x=x_train[:, 0],
        y=x_train[:, 1],
        mode='markers',
        marker=dict(color=y_train, colorscale='jet', opacity=0.7, size=8, symbol='circle'),
        name='Training Data',
        showlegend=True,
    )

    # Create a scatter plot of the test data
    scatter_test = go.Scatter(
        x=x_test[:, 0],
        y=x_test[:, 1],
        mode='markers',
        marker=dict(color=y_test, colorscale='jet', size=10, symbol='cross'),
        name='Testing Data',
        showlegend=True,
    )

    # Create a scatter plot for the legend (black color)
    legend_blue = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(color='rgb(142, 146, 203)', size=12, symbol='square'),
        name='Blue prediction boundary'
    )
    legend_red = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(color='rgb(188, 152, 159)', size=12, symbol='square'),
        name='Red prediction boundary'
    )

    # Create a contour plot of the decision boundary
    contour = go.Contour(
        x=np.arange(x_min, x_max, step),
        y=np.arange(y_min, y_max, step),
        z=preds_mesh,
        colorscale='jet',
        opacity=0.3,
        showscale=False,
        name='Decision Boundary',
        hovertemplate='x: %{x:.1f}<br>y: %{y:.1f}<br>Prediction: %{z:.0f}<extra></extra>',
    )

    # Calculate the accuracy of the model
    accuracy = round(np.mean(np.equal(y_test, preds)) * 100, 1)

    # Create the figure and add the plots
    fig = go.Figure(data=[scatter_train, scatter_test, legend_blue, legend_red, contour])
    fig.update_layout(
        title=f'SVM Accuracy = {accuracy}%',
        xaxis=dict(title='Feature 1'),
        yaxis=dict(title='Feature 2'),
    )

    return fig
