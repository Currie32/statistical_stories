from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from dash import callback, dcc, html, Input, Output, register_page
from plotly.graph_objs._figure import Figure as plotly_figure
from scipy.spatial.distance import cosine


register_page(__name__, path="/cosine-similarity")


layout = html.Div(className='content', children=[
    html.H1(className='content-title', children='Cosine Similarity'),
    html.Div(
        className="resource-link",
        children=[html.A("Link to scipy", target="_blank", href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html")]
    ),
    html.H2(className='section-title', children='Overview'),
    html.Div(className='paragraph', children=[
        html.P("Once upon a time, there was a curious fellow named Dave. Dave was an avid collector of rare gems and minerals, and he loved spending his weekends exploring caves and mines to find unique specimens. One day, he stumbled upon an old book in the dusty corner of an antique shop, and it caught his eye."),
        html.P("The book turned out to be about mathematical techniques applied to various fields, including data analysis. Intrigued, Dave decided to delve into this world of numbers and equations. As he flipped through the pages, he stumbled upon a section that talked about cosine similarity, a metric used in data science to measure the similarity between two vectors."),
        html.P("Dave, being a gem collector, immediately thought about how he could apply this concept to his hobby. He imagined each gem he owned as a vector in a high-dimensional space, where each dimension represented a different characteristic of the gem – like color, hardness, and transparency."),
        html.P("To understand cosine similarity, Dave imagined two gems as vectors in this space. The angle between these vectors would represent how similar or dissimilar the gems were in terms of their characteristics. The smaller the angle, the more similar the gems, and the larger the angle, the more dissimilar."),
        html.P("Excited to put his newfound knowledge into practice, Dave decided to compare two gems from his collection – a dazzling diamond and a vibrant emerald. He carefully noted down the characteristics of each gem and calculated the cosine similarity between their vectors."),
        html.P("As Dave crunched the numbers, he realized that the cosine similarity was high, indicating that the diamond and emerald shared many similar characteristics. This meant that, despite their differences, they were more alike than he initially thought."),
        html.P("Dave's journey into the world of cosine similarity not only enhanced his understanding of data science but also added a new layer to his gem-collecting hobby. From that day on, whenever he acquired a new gem, he couldn't resist calculating its cosine similarity to others in his collection, turning his passion into a fascinating blend of rocks and mathematics."),
    ]),
    html.H2(className='section-title', children='Summary'),
    html.Div(className='paragraph', children=[
        html.P("Cosine similarity is a measure that indicates how similar two vectors are by calculating the cosine of the angle between them.")
    ]),
    html.H2(className='section-title', children='Visualizing NDCG'),
    html.Div(className='plot-parameters', children=[
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='X (Vector A)'),
            dcc.Input(className='parameter-value', id='input-x-vector-a', value=16, min=-100, max=100, step=1, type='number'),
            html.Label(className='parameter-label', children='X (Vector B)'),
            dcc.Input(className='parameter-value', id='input-x-vector-b', value=-29, min=-100, max=100, step=1, type='number'),
        ]),
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='Y (Vector A)'),
            dcc.Input(className='parameter-value', id='input-y-vector-a', value=62, min=-100, max=100, step=1, type='number'),
            html.Label(className='parameter-label', children='Y (Vector B)'),
            dcc.Input(className='parameter-value', id='input-y-vector-b', value=37, min=-100, max=100, step=1, type='number'),
        ]),
    ]),
    html.Div(className='plot-full-width', children=[dcc.Graph(id='plot-cosine-similarity')]),
    html.H2(className='section-title', children='Formula'),
    html.Div(className='paragraph', children=[
        dcc.Markdown(
            r"""

            $$Cosine Similarity = 1 - \frac{{A \cdot B}}{{||A|| \cdot ||B||}}{}$$

            $$A \cdot B = a_1 \cdot b_1 + a_2 \cdot b_2 + ... + a_n \cdot b_n$$

            $$||A|| = \sqrt{A_1^2 + A_2^2 + ... + A_n^2}$$

            $$||B|| = \sqrt{B_1^2 + B_2^2 + ... + B_n^2}$$

            Where:
            - $$A \cdot B$$: The dot product of vectors A and B. The dot product is calculated by multiplying the corresponding elements of the two vectors and summing up the results.
            - $$||A||$$: ∥A∥: The Euclidean norm (magnitude) of vector A. It is calculated by taking the square root of the sum of the squares of each element in the vector.
            - $$||B||$$: The Euclidean norm of vector B.
            """,
            mathjax=True
        )
    ]),
    html.H2(className='section-title', children='Assumptions'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("1. Vector Representation: "), "The compared entities are appropriately represented as vectors. "]),
        html.P(children=[html.Strong("2. Independence of Dimensions: "), "Each dimension in the vector space is independent of the others. This assumption allows cosine similarity to work well in high-dimensional spaces."]),
        html.P(children=[html.Strong("3. Normalization: "), "The vectors involved in cosine similarity calculations are typically normalized. Normalization ensures that the magnitude or length of the vectors doesn't disproportionately influence the similarity measure. Normalized vectors allow the cosine similarity to focus on the direction of the vectors rather than their magnitudes."]),
        html.P(children=[html.Strong("4. Linearity: "), "Cosine similarity assumes linearity in the relationship between the features or dimensions of the vectors. If the relationship between dimensions is highly nonlinear, cosine similarity may not accurately capture the similarity between vectors."]),
        html.P(children=[html.Strong("5. Equal Importance of Features: "), "All dimensions or features contribute equally to the similarity calculation."]),
    ]),
    html.H2(className='section-title', children='Example Use Cases'),
    html.Div(className='paragraph', children=[
        html.P("1. To measure the similarity between documents, which can allow for document clustering, information retrieval, and plagiarism detection."),
        html.P("2. To recommend items to users based on the similarity of their preferences, e.g. movie recommendations, product recommendations, and content recommendations."),
        html.P("3. To identify similarities or differences in gene expression profiles across different conditions or treatments."),
        html.P("4. To compare molecular structures and identify potential drug candidates with similar chemical properties."),
        html.P("5. To identify anomalous patterns in transaction data, allowing for the detection of potentially fraudulent activities."),
    ]),
])


@callback(
    Output('plot-cosine-similarity', 'figure'),
    Input('input-x-vector-a', 'value'),
    Input('input-y-vector-a', 'value'),
    Input('input-x-vector-b', 'value'),
    Input('input-y-vector-b', 'value'),
)
def cosine_similarity(
    x_vector_a: int,
    y_vector_a: int,
    x_vector_b: int,
    y_vector_b: int,
) -> plotly_figure:
    """
    Calculate the cosine similarity between two vectors and plot them.
    """
    # Define two vectors
    vector_a = np.array([x_vector_a, y_vector_a])
    vector_b = np.array([x_vector_b, y_vector_b])

    # Calculate cosine similarity
    cosine_similarity = round(1 - cosine(vector_a, vector_b), 4)

    # Create vectors as arrows
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0, vector_a[0]],
        y=[0, vector_a[1]],
        mode='lines+text',
        line=dict(color='blue', width=2),
        marker=dict(color='blue'),
        text=['', 'Vector A'],
        textposition='top left'
    ))

    fig.add_trace(go.Scatter(
        x=[0, vector_b[0]],
        y=[0, vector_b[1]],
        mode='lines+text',
        line=dict(color='green', width=2),
        marker=dict(color='green'),
        text=['', 'Vector B'],
        textposition='top right'
    ))

    # Set layout
    fig.update_layout(
        title=f'Cosine Similarity = {cosine_similarity}',
        xaxis=dict(title='X', range=[-100, 100]),
        yaxis=dict(title='Y', range=[-100, 100]),
        showlegend=False
    )

    return fig