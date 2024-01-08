from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, register_page, Input, Output, callback
from plotly.graph_objs._figure import Figure as plotly_figure
from scipy.stats import bernoulli


register_page(__name__, path="/bernoulli")

layout = html.Div(className='content', children=[
    html.H1(className='content-title', children='Bernoulli Distribution'),
    html.Div(
        className="resource-link",
        children=[html.A("Link to scipy", target="_blank", href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html")]
    ),
    html.H2(className='section-title', children='Overview'),
    html.Div(className='paragraph', children=[
        html.P("Once upon a time, there lived a curious man named Dave, an enthusiastic fisherman who spent his weekends by the local lake. Little did he know, his fishing adventures would lead him to unravel the mysteries of the Bernoulli distribution."),
        html.P("One sunny day, as Dave cast his fishing line into the calm waters, he noticed something peculiar about his catches. He observed that some days he would effortlessly catch a fish, while on other days, it seemed like the fish were playing hard to get."),
        html.P("Intrigued by this pattern, Dave decided to keep a record of his catches. He noted down whether he caught a fish (success) or not (failure) each day. As he accumulated more data, a pattern emerged – a binary outcome of success or failure."),
        html.P("Driven by curiosity, Dave sought information about his observations. He discovered that his fishing adventures perfectly mirrored the concept of a Bernoulli distribution."),
        html.P("The Bernoulli distribution, he learned, models random experiments with two possible outcomes: success and failure. Catching a fish represented success, while returning home empty-handed denoted failure."),
        html.P("Dave understood that each fishing day was an independent trial, meaning the outcome of one day did not influence the outcome of another."),
        html.P("Armed with this newfound knowledge, Dave realized that the Bernoulli distribution was a fantastic tool for describing the probability of success or failure in a single, independent trial. He began predicting the likelihood of a successful fishing day based on his historical data."),
    ]),
    html.H2(className='section-title', children='Summary'),
    html.Div(className='paragraph', children=[
        html.P("The Bernoulli distribution models the probability of success or failure in a single trial of a binary experiment, where success is assigned the value 1 and failure is assigned the value 0.")
    ]),
    html.H2(className='section-title', children='Visualizing the Bernoulli Distribution'),
    html.Div(className='plot-parameters', children=[
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='Probability of Success'),
            dcc.Input(className='parameter-value', id='input-p-bernoulli', value=0.32, min=0.01, max=1.0, step=0.01, type='number'),
        ]),
    ]),
    html.Div(className='plots-two', children=[
        html.Div(className='plot', children=[dcc.Graph(id='plot-pmf-bernoulli')]),
        html.Div(className='plot', children=[dcc.Graph(id='plot-cdf-bernoulli')])
    ]),
    html.H2(className='section-title', children='Assumptions'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("1. Binary Outcomes: "), "The experiment should have only two possible outcomes, often denoted as success and failure."]),
        html.P(children=[html.Strong("2. Independent Trials: "), "The outcome of one trial should not influence the outcome of another."]),
        html.P(children=[html.Strong("3. Constant Probability of Success: "), "The probability of success remains constant across all trials."]),
    ]),
    html.H2(className='section-title', children='Formula'),
    html.Div(className='paragraph', children=[
        dcc.Markdown(
            r"""
            $$f(x) = p^x \cdot (1 - p)^{1 - x}$$

            Where:
            - $$x$$: The random variable representing the outcome of a single Bernoulli trial. It takes on the value 1 if the success event occurs and 0 if the failure event occurs.
            - $$p$$: The probability of success.
            - $$1 - p$$: The probability of failure.
            """,
            mathjax=True
        )
    ]),
    html.H2(className='section-title', children='Examples'),
    html.Div(className='paragraph', children=[
        html.P(children=["1. To model the probability of a stock increasing or decreasing in price."]),
        html.P(children=["2. To estimate the probability of a user clicking on an ad on a webpage."]),
        html.P(children=["3. To predict the success or failure of a medical treatment."]),
        html.P(children=["4. To model the probability of it raining tomorrow."]),
        html.P(children=["5. To assess the likelihood of an insurance claim being filed."]),
    ])
])


@callback(
    Output('plot-pmf-bernoulli', 'figure'),
    Output('plot-cdf-bernoulli', 'figure'),
    Input('input-p-bernoulli', 'value'),
)
def pmf_cdf_beta(
    p: float,
) -> Tuple[plotly_figure, plotly_figure]:
    """
    Plot the probability mass function for a Bernoulli distribution
    and its cumulative distribution function.
    """

    x = np.linspace(-0.05, 1.05, 100)

    y_pmf = bernoulli.pmf([0, 0.5, 1], p)
    fig_pmf = go.Figure(
        data=go.Bar(
            x=[0, 0.5, 1],
            y=y_pmf,
            hovertemplate='x: %{x:.2f}<br>Probability of Success: %{y:.2f}<extra></extra>'
        )
    )
    fig_pmf.update_layout(
        title=dict(
            text='Probability Mass Function',
            x=0.5,
        ),
        xaxis=dict(title='X'),
        yaxis=dict(title='Probability of Success'),
    )

    y_cdf = bernoulli.cdf(x, p)
    fig_cdf = go.Figure(
        data=go.Scatter(
            x=x,
            y=y_cdf,
            hovertemplate='x: %{x:.2f}<br>Probability of Success: %{y:.2f}<extra></extra>'
        )
    )
    fig_cdf.update_layout(
        title=dict(
            text='Cumulative Distribution Function',
            x=0.5,
        ),
        xaxis=dict(title='X'),
        yaxis=dict(title='Probability of Success'),
    )

    return fig_pmf, fig_cdf
