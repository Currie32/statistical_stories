from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from dash import callback, dcc, html, Input, Output, register_page
from plotly.graph_objs._figure import Figure as plotly_figure
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


register_page(__name__, path="/k-nearest-neighbors")

layout = html.Div(className='content', children=[
    html.H1(className='content-title', children='k-Nearest Neighbors'),
    html.Div(
        className="resource-link",
        children=[html.A("Link to scikit-learn", target="_blank", href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html")]
    ),
    html.H2(className='section-title', children='Overview'),
    html.Div(className='paragraph', children=[
        html.P("Once upon a time, there was a curious fellow named Dave who loved spending his time exploring nature. Dave had a special interest in trees and wanted to learn how to identify different types. He had heard about a powerful algorithm called k-nearest neighbors (KNN) that could predict things based on their similarity to other things. Intrigued, Dave decided to test it out."),
        html.P("Dave began his quest by gathering information about various trees found in his favourite forest. He noted down features such as the tree's height, type of leaves, bark texture, and colour of flowers (if any). Now with plenty of data, Dave set out to apply the k-nearest neighbors algorithm to his tree classification project."),
        html.P("To begin, Dave visualized all the trees on a plot, assigning each tree a point based on its features. For instance, a tall tree with wide leaves and smooth bark would be represented as a point on the plot. Similarly, he plotted other trees like oak, pine, and maple, each with its set of features."),
        html.P('With the trees scattered across the plot, it was time for Dave to test out the k-nearest neighbors algorithm. The value of "k" would determine the number of most similar trees Dave considered when making predictions.'),
        html.P("On Dave’s next hike through a forest, he stumbled upon a magnificent tree in the woods, unlike any he had encountered before. He carefully observed its features and plotted a new point on the plot to represent this mysterious tree."),
        html.P("Curious to know the tree's identity, Dave examined the k nearest points to the new tree's point. These nearest points represented trees that were most similar to the one he had discovered. If the majority of the nearest points belonged to oak trees, then it was highly likely that the new tree was an oak as well."),
        html.P("Excited by the algorithm's potential, Dave decided to put it to the test. He set the value of k to 5 and identified the five nearest points to the new tree. Among them, three were oak trees, one was a pine, and one was a maple. Considering the majority, Dave predicted that the new tree was most likely an oak."),
        html.P("Dave's journey through the forest continued, with him encountering diverse trees and employing the k-nearest neighbors algorithm to make predictions. Sometimes the algorithm proved remarkably accurate, while other times it made mistakes and required more data or an adjustment to the value of k to be correct. Dave understood that the quality and quantity of data played a crucial role in the accuracy of predictions."),
        html.P("Through his tree-tastic adventures and experiments, Dave realized the power of the k-nearest neighbors algorithm in identifying and classifying trees based on their similarities. It allowed him to delve deeper into the enchanting world of trees, appreciating their unique features and learning about their various species."),
        html.P("Grateful for this newfound knowledge, Dave continued to explore nearby lands, employing the algorithm in different ways to solve tree-related mysteries and foster his profound love for nature. And so, his journey of discovery and admiration flourished, all thanks to the enchanting capabilities of the k-nearest neighbors algorithm."),
    ]),
    html.H2(className='section-title', children='Visualizing the k-nearest neighbors algorithm'),
    html.Div(className='paragraph-italic', children=[
        html.P(children=[
            "The algorithm's parameters are defined on the ",
            html.A("scikit-learn website.", target="_blank", href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html")
        ])
    ]),
    html.Button('Generate New Data', id='button-new-data', n_clicks=0),
    html.Div(className='plot-parameters', children=[
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='n_neighbors'),
            dcc.Input(className='parameter-value', id='input-n-neighbors-knn', value=5, min=1, max=250, step=1, type='number'),
        ]),
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='weights'),
            dcc.Dropdown(
                id='input-weights-knn',
                value='uniform',
                options=[
                    {'label': 'uniform', 'value': 'uniform'},
                    {'label': 'distance', 'value': 'distance'},
                ],
                clearable=False,
            ),
        ]),
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='algorithm'),
            dcc.Dropdown(
                id='input-algorithm-knn',
                value='auto',
                options=[
                    {'label': 'auto', 'value': 'auto'},
                    {'label': 'ball_tree', 'value': 'ball_tree'},
                    {'label': 'brute', 'value': 'brute'},
                ],
                clearable=False,
            ),
        ]),
    ]),
    html.Div(className='plot-full-width', children=[dcc.Graph(id='plot-knn')]),
    html.H2(className='section-title', children='Assumptions'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("1. Similarity: "), "Data points within close proximity in the feature space are likely to belong to the same class or have similar labels."]),
        html.P(children=[html.Strong("2. Feature scaling: "), "Features are on similar scales, so it is important to normalize or scale the data to prevent certain features from dominating the distance calculation due to their larger ranges or magnitudes. Normalizing the features ensures that they contribute proportionally to the similarity measurement."]),
        html.P(children=[html.Strong("3. Symmetry: "), "The data is symmetrically distributed around its mean. The left and right sides of the distribution are mirror images of each other. If the data is skewed (asymmetric), the normal distribution may not accurately represent the underlying distribution."]),
        html.P(children=[html.Strong("4: Independence of features: "), "The features used to calculate similarity are independent of each other. Violating this assumption, such as in the case of highly correlated features, can lead to biased predictions and inaccurate results."]),
        html.P(children=[html.Strong("5: Balanced Class Distribution: "), "The class distribution is relatively balanced, otherwise this can result in biased predictions where the majority class dominates the predictions and the minority class(es) is overlooked."]),
    ]),
    html.H2(className='section-title', children='Appropriate use cases'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("Classification: "), "Assigning instances to predefined classes, including binary and multi-class problems."]),
        html.P(children=[html.Strong("Explainability: "), "Due to the algorithm's simplicity, it can be advantageous when the explainability of predictions is important."]),
        html.P(children=[html.Strong("Recommendation Systems: "), "Providing personalized recommendations based on the similarity between users or items."]),
        html.P(children=[html.Strong("Anomaly Detection: "), "Identifing unusual or anomalous instances in a dataset. The KNN algorithm can be applied to measure the distance or dissimilarity between instances and identify those that deviate significantly from the majority."]),
        html.P(children=[html.Strong("Imputation of Missing Values: "), "Imputing missing values in datasets, especially when the missing values are expected to have similar values to their neighbors."]),
    ]),
    html.H2(className='section-title', children='Inappropriate use cases'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("Large Datasets: "), "Performance can drop, especially if the dimensionality of the data is high. As the number of instances and features increases, the computational cost of finding nearest neighbors grows significantly. In such cases, algorithms like decision trees, random forests, or gradient boosting techniques could offer better scalability and efficiency."]),
        html.P(children=[html.Strong("High-Dimensional Data: "), "When the number of features is large relative to the number of instances, the KNN algorithm can suffer from the curse of dimensionality. As the number of dimensions increases, the sparsity of the data increases, and the concept of proximity becomes less meaningful. Dimensionality reduction techniques like principal component analysis (PCA) or manifold learning methods, followed by the application of other algorithms, may be more effective in such scenarios. There isn't a minimum ratio of data points to features to avoid dimensionality issues, but at least 5-10 is a good rule of thumb. As problems become more complex, a higher ratio is beneficial."]),
        html.P(children=[html.Strong("Imbalanced Datasets: "), "Where one or more classes are significantly underrepresented, the KNN algorithm might produce biased predictions. This is because the majority class can dominate the nearest neighbor selection. In imbalanced datasets, algorithms specifically designed for handling class imbalance, such as ensemble methods with class weighting or resampling techniques, may yield better results."]),
        html.P(children=[html.Strong("Non-Uniform Attribute Relevance: "), "All features are assumed to have equal relevance in determining similarity. However, if certain features are more important than others, using an algorithm that incorporates feature selection or feature weighting techniques, such as linear models or decision trees with feature importance measures, might provide better predictive performance."]),
        html.P(children=[html.Strong("Non-Linear Relationships: "), "If the relationship between features and classes is highly non-linear, the KNN algorithm may struggle to capture complex patterns. In such cases, algorithms like support vector machines (SVMs), neural networks, or kernel-based methods could be more suitable."]),
    ]),
])


@callback(
    Output('plot-knn', 'figure'),
    Input('input-n-neighbors-knn', 'value'),
    Input('input-weights-knn', 'value'),
    Input('input-algorithm-knn', 'value'),
    Input('button-new-data', 'n_clicks'),
)
def k_nearest_neighbors(
    n_neighbors: int,
    weights: str,
    algorithm: str,
    random_state: int,
) -> Tuple[plotly_figure, plotly_figure]:
    """
    Generate training and testing data for a KNN model.
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

    # Fit the KNN classifier on the training data
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
    knn.fit(x_train, y_train)
    preds = knn.predict(x_test)

    # Generate a meshgrid to visualize the decision boundary
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))

    # Predict the class labels for each point in the meshgrid
    preds_mesh = knn.predict(np.c_[xx.ravel(), yy.ravel()])
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
        title=f'K-Nearest Neighbors (KNN) Algorithm, Accuracy = {accuracy}%',
        xaxis=dict(title='Feature 1'),
        yaxis=dict(title='Feature 2'),
    )

    return fig
