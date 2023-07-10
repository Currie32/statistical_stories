from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, register_page, Input, Output, callback
from plotly.graph_objs._figure import Figure as plotly_figure
from scipy.stats import norm


register_page(__name__, path="/normal")

layout = html.Div(className='content', children=[
    html.H1(className='content-title', children='Normal Distribution'),
    html.Div(
        className="resource-link",
        children=[html.A("Link to numpy", target="_blank", href="https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html")]
    ),
    html.H2(className='section-title', children='Overview'),
    html.Div(className='paragraph', children=[
        html.P("Once upon a time there was a farmer named Dave. Dave had a big field where he grew his crops, and he was always curious about how his crops grew. He wanted to understand how the height of his plants varied from one another."),
        html.P("One day, Dave decided to measure the height of all the cornstalks in his field. He found that most of the cornstalks were of average height, neither too tall nor too short. However, there were a few cornstalks that were very tall, and a few that were very short."),
        html.P("Intrigued by this observation, Dave decided to plot a graph to represent the heights of his cornstalks. He made a horizontal line and marked different heights of the cornstalks along it. On the vertical axis, he marked the number of cornstalks with each height."),
        html.P("As Dave plotted the graph, he noticed something interesting. The graph had a bell-shaped curve! Most of the cornstalks were in the middle, around the average height, forming the highest point of the curve. As he moved away from the average height, the number of cornstalks decreased, forming the sloping sides of the curve."),
        html.P("Dave was fascinated by this bell-shaped curve, and he learned that it was called the normal distribution (also known as the Gaussian Distribution). He discovered that the normal distribution is a mathematical pattern that often appears in nature and human-made systems."),
        html.P("Dave also learned that the normal distribution has some unique properties. He found out that the average height of the cornstalks was right in the middle of the curve, called the mean. He also learned that the curve was symmetrical, meaning that the left and right sides were mirror images of each other."),
        html.P("Furthermore, Dave discovered that the normal distribution had a standard deviation. The standard deviation is a measure of how spread out the data is. If the standard deviation is small, the data points are closely packed together, creating a narrow curve. If the standard deviation is large, the data points are more spread out, making the curve wider."),
        html.P("With this newfound knowledge, Dave was able to make predictions about his cornfield. He could estimate how many cornstalks would fall within a certain height range based on the properties of the normal distribution."),
    ]),
    html.H2(className='section-title', children='Visualizing the Normal Distribution'),
    html.Div(className='plot-parameters', children=[
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='Mean'),
            dcc.Input(className='parameter-value', id='mean-input-normal', value=0, min=-1000, max=1000, step=0.1, type='number'),
        ]),
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='Standard Deviation'),
            dcc.Input(className='parameter-value', id='std-input-normal', value=1, min=-1000, max=1000, step=0.1, type='number'),
        ])
    ]),
    html.Div(className='plots-two', children=[
        html.Div(className='plot', children=[dcc.Graph(id='plot-pdf-normal')]),
        html.Div(className='plot', children=[dcc.Graph(id='plot-cdf-normal')])
    ]),
    html.H2(className='section-title', children='Assumptions'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("1. Independence: "), "The data points should be independent of each other, i.e. one observation should not be influenced or dependent on another observation."]),
        html.P(children=[html.Strong("2. Continuous Data: "), "The data should be continuous, meaning it can take any value within a range. Discrete or categorical data may not follow a normal distribution, and alternative distribution models may be more appropriate."]),
        html.P(children=[html.Strong("3. Symmetry: "), "The data is symmetrically distributed around its mean. The left and right sides of the distribution are mirror images of each other. If the data is skewed (asymmetric), the normal distribution may not accurately represent the underlying distribution."]),
        html.P(children=[html.Strong("4: Constant Variance: "), "The variance of the data, which measures how spread out the observations are, is assumed to be constant across the entire range of values. This is known as homoscedasticity. If the variance is not constant (heteroscedasticity), the normal distribution may not be appropriate"]),
        html.P(children=[html.Strong("5: Outliers: "), "The normal distribution assumes that the data is free from outliers, which are extreme values that are far from the bulk of the data. Outliers can significantly affect the mean and standard deviation, potentially distorting the normal distribution."]),
    ]),
    html.H2(className='section-title', children='Formula'),
    html.Div(className='paragraph', children=[
        dcc.Markdown(
            r"""
            $$f(x) = \frac{1}{{\sigma \sqrt{{2\pi}}}} \exp\left(-\frac{{(x - \mu)^2}}{{2\sigma^2}}\right)$$

            Where:
            - $$f(x)$$: The probability density function, which gives the relative likelihood of observing a specific value (x) in the distribution.
            - $x$: The random variable, which is a particular value within the distribution for which we want to calculate the probability.
            - $μ$ $(mu)$: The mean of the distribution. It determines the center or average value around which the data is symmetrically distributed.
            - $σ$ $(sigma)$: The standard deviation of the distribution. It quantifies the spread or dispersion of the data points around the mean. A larger value of σ indicates a wider spread, while a smaller value indicates a narrower spread.
            - $exp()$: The exponential function, which raises the mathematical constant e (approximately 2.71828) to the power of the value within the parentheses.
            - $π$ $(pi)$: The mathematical constant approximately equal to 3.14159 and represents the ratio of the circumference of a circle to its diameter.
            - $√(2π)$: The square root of 2π, which is a normalization factor that ensures the total area under the curve sums to 1. It standardizes the probability density function.
            """,
            mathjax=True
        )
    ]),
    html.H2(className='section-title', children='Examples'),
    html.Div(className='paragraph', children=[
        html.P(children=["1. Human heights and weights."]),
        html.P(children=["2. Test scores of students on standardized tests."]),
        html.P(children=["3. Returns of stocks and assets over a specific period of time."]),
        html.P(children=["4. Certain characteristics of products, such as weight, length, or volume in manufacturing and quality control processes."]),
        html.P(children=["5. Ratings in customer satisfaction surveys."]),
        html.P(children=["6. Measurement errors in scientific experiments or engineering."]),
        html.P(children=["7. Various time-related events, such as the duration of phone calls, waiting times, or service times."]),
    ]),
    html.H2(className='section-title', children='Code'),
    html.Div(className='paragraph', children=[
        html.P(children=["This plot can be generated using the code below it."]),
    ]),
    html.Button('Generate New Data', id='button-new-data', n_clicks=0),
    dcc.Graph(id='plot-histogram-normal'),
    html.Div(className='paragraph', children=[
        html.Pre(
            '''
import numpy as np
import plotly.graph_objects as go


# Set the parameters for sampling from a normal distribution
mu = 0  # Mean
sigma = 1  # Standard deviation
samples = 1000

sample = np.random.normal(mu, sigma, samples)

histogram = go.Histogram(
    x=sample,
    histnorm='probability density',
    opacity=0.7,
    marker=dict(color='lightblue', line=dict(color='darkblue', width=1)),
    showlegend=False,
    hovertemplate='x: %{x:.1f}<br>Probability Density: %{y:.2f}<extra></extra>',
)

layout = go.Layout(
    title='Normal Distribution',
    xaxis=dict(title='Value', range=[-4, 4]),
    yaxis=dict(title='Probability Density'),
    showlegend=True,
)

# Create the plot
go.Figure(data=[histogram], layout=layout)
            ''')
    ])
])


@callback(
    Output('plot-pdf-normal', 'figure'),
    Output('plot-cdf-normal', 'figure'),
    Input('mean-input-normal', 'value'),
    Input('std-input-normal', 'value'),
)
def pdf_cdf_normal(
    mean: float,
    std: float,
) -> Tuple[plotly_figure, plotly_figure]:
    """
    Plot the probability density function for a normal distribution
    and its cumulative density function.
    """

    # Only show x-axis values between -10 and 10, using 200 steps
    x = np.linspace(-10, 10, 200)

    y_pdf = norm.pdf(x, loc=mean, scale=std)
    fig_pdf = go.Figure(
        data=go.Scatter(
            x=x,
            y=y_pdf,
            hovertemplate='x: %{x:.1f}<br>Probability Density: %{y:.2f}<extra></extra>'
        )
    )
    fig_pdf.update_layout(
        title=dict(
            text='Probability Density Function',
            x=0.5,
        ),
        xaxis=dict(title='X'),
        yaxis=dict(title='Probability Density'),
    )

    y_cdf = norm.cdf(x, loc=mean, scale=std)
    fig_cdf = go.Figure(
        data=go.Scatter(
            x=x,
            y=y_cdf,
            hovertemplate='x: %{x:.1f}<br>Probability Density: %{y:.2f}<extra></extra>'
        )
    )
    fig_cdf.update_layout(
        title=dict(
            text='Cumulative Density Function',
            x=0.5,
        ),
        xaxis=dict(title='X'),
        yaxis=dict(title='Probability Density'),
    )

    return fig_pdf, fig_cdf


@callback(
    Output('plot-histogram-normal', 'figure'),
    Input('button-new-data', 'n_clicks'),
)
def histogram_normal(n_clicks: int) -> plotly_figure:
    """
    Sample from a normal distribution, then generate a histogram
    using this data. Update the plot each time "button-new-data" is clicked.
    """
    
    # Set the mean and standard deviation of the normal distribution
    mu = 0  # Mean
    sigma = 1  # Standard deviation
    samples = 1000

    sample = np.random.normal(mu, sigma, samples)

    histogram = go.Histogram(
        x=sample,
        histnorm='probability density',
        opacity=0.7,
        marker=dict(color='lightblue', line=dict(color='darkblue', width=1)),
        showlegend=False,
        hovertemplate='x: %{x:.1f}<br>Probability Density: %{y:.2f}<extra></extra>',
    )

    layout = go.Layout(
        title='Normal Distribution',
        xaxis=dict(title='X', range=[-4, 4]),
        yaxis=dict(title='Probability Density'),
        showlegend=True,
    )

    fig = go.Figure(data=[histogram], layout=layout)

    return fig
