from typing import Tuple

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from plotly.graph_objs._figure import Figure as plotly_figure
from scipy.stats import norm, poisson, shapiro


layout_poisson = html.Div(className='content', children=[
    html.H1(className='content-title', children='Poisson Distribution'),
    html.Div(
        className="resource-link",
        children=[html.A("Link to numpy", target="_blank", href="https://numpy.org/doc/stable/reference/random/generated/numpy.random.poisson.html")]
    ),
    html.H2(className='section-title', children='Overview'),
    html.Div(className='paragraph', children=[
        html.P("Once upon a time there was a bakery run by a friendly baker named Dave. After years of baking, Dave noticed that during the morning rush the number of croissants that he would sell could change every hour."),
        html.P("Curious to discover a pattern, Dave started keeping track of the number of croissants he sold during each hour. He recorded the data for several days and noticed something fascinating. Some hours he sold exactly 10 croissants, while in other hours he sold slightly more or slightly fewer."),
        html.P("Dave wondered if there was a way to predict the number of croissants he would sell in a given hour. He visited his friend, Dan, who was a mathematician, and shared his data with him."),
        html.P('Dan smiled and said, “Dave, what you\'re experiencing is an example of the Poisson distribution." He explained that the Poisson distribution helps us understand the number of events that occur over a fixed period of time or space.'),
        html.P('Dan continued, "The Poisson distribution tells us the probability of selling a certain number of croissants in an hour. The average number of croissants you sell per hour is called the lambda (λ). In your case, λ is 10 since that\'s the average number of croissants you sell during the morning rush."'),
        html.P('Dan added, “Another interesting quality of the Poisson distribution is that the variance (σ^2) is equal to the average. This means that if you were to sell more croissants every hour during the morning rush, the variability of the number of croissants sold would also increase.”'),
        html.P('Dave nodded, starting to grasp the concept. He replied, "So I can use this distribution to help me predict the number of croissants I\'ll sell.”'),
        html.P('Dan exclaimed, “Exactly! The Poisson distribution gives you the probability mass function, which tells you the likelihood of selling a specific number of croissants. For example, if we want to know the probability of selling 8 croissants in an hour, we can use the Poisson distribution formula to calculate it."'),
        html.P("Dave was eager to try it out. Dan taught him the formula and they worked together to calculate the probabilities for various numbers of croissants sold in an hour. Dave realized that the distribution allowed him to estimate the likelihood of selling a particular number of croissants, even if the actual sales varied from hour to hour."),
        html.P("Armed with this knowledge, Dave felt more prepared to handle the fluctuations in his bakery's sales. He continued to use the Poisson distribution as a tool to guide his production planning and make informed decisions about how many croissants to bake each morning."),
        html.P("And so, Dave's bakery flourished as he embraced the Poisson distribution, bringing his customers a steady supply of delicious croissants, while also satisfying his own curiosity about the patterns in his sales data."),
    ]),
    html.H2(className='section-title', children='Visualizing the Poisson Distribution'),
    html.Div(className='plot-parameters', children=[
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='Lambda'),
            dcc.Input(className='parameter-value', id='poisson-lambda-input', value=1, min=0, max=1000, step=0.1, type='number'),
        ]),
    ]),
    html.Div(className='plots-distribution', children=[
        html.Div(className='plot', children=[dcc.Graph(id='poisson-plot')]),
        html.Div(className='plot', children=[dcc.Graph(id='poisson-cdf-plot')])
    ]),
    html.H2(className='section-title', children='Assumptions'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("1. Independence: "), "The data points should be independent of each other, i.e. one observation should not be influenced or dependent on another observation."]),
        html.P(children=[html.Strong("2. Integers: "), "The data should be integers, meaning it can take values such as 0, 1, 2, etc."]),
        html.P(children=[html.Strong("3. Constant Rate: "), "The average rate at which events occur should remain constant over the given time interval or within a defined space. This average rate is denoted by λ (lambda), which represents the mean number of events in the specified time or space."]),
        html.P(children=[html.Strong("4: Asynchronous: "), "Two events cannot occur at the same time or in the same location."]),
    ]),
    html.H2(className='section-title', children='Formula'),
    html.Div(className='paragraph', children=[
        dcc.Markdown(
            r"""
            $$f(k) = \frac{{e^{-\lambda} \cdot \lambda^k}}{{k!}}$$

            Where:
            - $$f(k)$$: The probability mass function (PMF), which gives the probability of the random variable having the value k.
            - $$e$$: Euler's number, approximately equal to 2.71828, and it serves as the base of the natural logarithm.
            - $$λ$$ $$(lambda)$$: The average rate or expected number of events occurring in a specific time or space interval. It is also the mean of the Poisson distribution.
            - $$k$$: The number of events for which we want to find the probability. It can take any non-negative integer value.
            - $$k!$$: The factorial of k, which is the product of all positive integers less than or equal to k.
            """,
            mathjax=True
        )
    ]),
    html.H2(className='section-title', children='Examples'),
    html.Div(className='paragraph', children=[
        html.P(children=["1. The arrival of phone calls at a call center or the number of emails received per minute."]),
        html.P(children=["2. The number of defects or errors that occurred during a production process."]),
        html.P(children=["3. The occurrence of insurance claims, such as car accidents or property damage."]),
        html.P(children=["4. The number of customers arriving at a store during a specific time period."]),
        html.P(children=["5. The frequency of certain medical events, such as hospital admissions for a specific condition."]),
        html.P(children=["6. The number of website visits or requests received per unit of time."]),
        html.P(children=["7. The occurrence of natural events like earthquakes or volcanic eruptions in a specific region."]),
    ]),
    html.H2(className='section-title', children='Code'),
    html.Div(className='paragraph', children=[
        html.P(children=["This plot can be generated using the code below it."]),
    ]),
    html.Button('Generate New Data', id='button-new-data', n_clicks=0),
    dcc.Graph(id='histogram-plot'),
    html.Div(className='paragraph', children=[
        html.Pre(
            '''
import numpy as np
import plotly.graph_objects as go


# Set the parameters for sampling from a poisson distribution
lmbda = 2.0
samples = 1000

sample = np.random.poisson(lmbda, samples)

histogram = go.Histogram(
    x=sample,
    histnorm='probability density',
    opacity=0.7,
    marker=dict(color='lightblue', line=dict(color='darkblue', width=1)),
    showlegend=False,
    hovertemplate='x: %{x:.0f}<br>Probability Density: %{y:.2f}<extra></extra>',
)

layout = go.Layout(
    title='Poisson Distribution',
    xaxis=dict(title='Value', range=[-4, 10]),
    yaxis=dict(title='Probability Density'),
    showlegend=True,
)

# Create the plot
go.Figure(data=[histogram], layout=layout)
            ''')
    ])
])


def distribution_plots_poisson(
    lam: float,
) -> Tuple[plotly_figure, plotly_figure]:
    """
    Update the poisson distributions plots using the new lambda value.
    """
    # Only show x-axis values between 0 and 25
    x = np.arange(0, 26)

    y_pdf = poisson.pmf(x, lam)
    fig_pdf = go.Figure(data=go.Scatter(x=x, y=y_pdf, hovertemplate='x: %{x:.0f}<br>Probability Density: %{y:.2f}<extra></extra>'))
    fig_pdf.update_layout(
        title=dict(
            text='Probability Mass Function',
            x=0.5,
        ),
        xaxis=dict(title='X'),
        yaxis=dict(title='Probability Density'),
    )

    y_cdf = poisson.cdf(x, lam)
    fig_cdf = go.Figure(data=go.Scatter(x=x, y=y_cdf, hovertemplate='x: %{x:.0f}<br>Probability Density: %{y:.2f}<extra></extra>'))
    fig_cdf.update_layout(
        title=dict(
            text='Cumulative Mass Function',
            x=0.5,
        ),
        xaxis=dict(title='X'),
        yaxis=dict(title='Probability Density'),
    )

    return fig_pdf, fig_cdf


def histogram_poisson_plot() -> go.Figure:
    """
    Sample from a Poisson distribution, then generate a histogram
    using this data.
    """
    
    # Set the parameters for sampling from a poisson distribution
    lmbda = 2.0
    samples = 1000

    sample = np.random.poisson(lmbda, samples)

    histogram = go.Histogram(
        x=sample,
        histnorm='probability density',
        opacity=0.7,
        marker=dict(color='lightblue', line=dict(color='darkblue', width=1)),
        showlegend=False,
        hovertemplate='x: %{x:.0f}<br>Probability Density: %{y:.2f}<extra></extra>',
    )

    layout = go.Layout(
        title='Poisson Distribution',
        xaxis=dict(title='X', range=[-4, 10]),
        yaxis=dict(title='Probability Density'),
        showlegend=True,
    )

    # Create the plot
    fig = go.Figure(data=[histogram], layout=layout)

    return fig
