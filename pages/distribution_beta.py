from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, register_page, Input, Output, callback
from plotly.graph_objs._figure import Figure as plotly_figure
from scipy.stats import beta


register_page(__name__, path="/beta")

layout = html.Div(className='content', children=[
    html.H1(className='content-title', children='Beta Distribution'),
    html.Div(
        className="resource-link",
        children=[html.A("Link to scipy", target="_blank", href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html")]
    ),
    html.H2(className='section-title', children='Overview'),
    html.Div(className='paragraph', children=[
        html.P("In a quaint small town, there was a passionate baker named Dave with dreams of perfecting his recipes. One day, while experimenting with different ingredients, he stumbled upon a statistical gem that would elevate his baking game - the beta distribution."),
        html.P("Curious and eager to understand the magic behind it, Dave began his journey into the world of probability. He learned that the beta distribution is a versatile tool, especially for modeling the distribution of random variables that represent proportions or percentages. Intrigued, he saw the potential to apply this to his baking adventures."),
        html.P("Dave decided to focus on perfecting his chocolate chip cookies. He thought about the varying proportions of ingredients and the impact they had on the final product. The beta distribution, with its range between 0 and 1, seemed like the perfect fit for capturing the probabilities associated with different ratios of flour, sugar, and chocolate chips."),
        html.P("Armed with this statistical insight, Dave meticulously measured and recorded the proportions of each ingredient in his cookie recipes. Using the beta distribution, he could predict the likelihood of achieving that perfect balance of chewiness and crispiness."),
        html.P("As Dave continued his baking experiments, he realized that the beta distribution was not just a mathematical concept but a powerful tool to optimize and tailor his recipes. It helped him understand the uncertainties and probabilities associated with ingredient proportions, enabling him to create a consistent and delightful batch of cookies every time."),
        html.P("Word spread about Dave's magical cookies, and soon, his small town was buzzing with excitement about the baker who used statistical wizardry to achieve cookie perfection. The beta distribution had transformed Dave's baking hobby into a precise and delectable art, leaving the town's residents craving more of his statistically inspired treats.")
    ]),
    html.H2(className='section-title', children='Summary'),
    html.Div(className='paragraph', children=[
        html.P("The beta distribution is a continuous probability distribution that models the likelihood of a random variable taking on values between 0 and 1, often used to represent proportions or probabilities.")
    ]),
    html.H2(className='section-title', children='Visualizing the Beta Distribution'),
    html.Div(className='plot-parameters', children=[
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='Alpha'),
            dcc.Input(className='parameter-value', id='input-alpha-beta', value=2.8, min=0.1, max=100, step=0.1, type='number'),
        ]),
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='Beta'),
            dcc.Input(className='parameter-value', id='input-beta-beta', value=1.5, min=0.1, max=100, step=0.1, type='number'),
        ])
    ]),
    html.Div(className='plots-two', children=[
        html.Div(className='plot', children=[dcc.Graph(id='plot-pdf-beta')]),
        html.Div(className='plot', children=[dcc.Graph(id='plot-cdf-beta')])
    ]),
    html.H2(className='section-title', children='Assumptions'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("1. Interval Constraints: "), "The distribution is applicable to random variables that are constrained within the interval [0, 1]. This aligns with proportions or percentages."]),
        html.P(children=[html.Strong("2. Independence of Observations: "), "Each data point or event is not influenced by the others."]),
        html.P(children=[html.Strong("3. Continuous Data: "), "designed for continuous data rather than discrete values."]),
    ]),
    html.H2(className='section-title', children='Formula'),
    html.Div(className='paragraph', children=[
        dcc.Markdown(
            r"""
            $$f(x;α,β) = \frac{x^{α - 1} \cdot (1 - x)^{β - 1}}{B(α, β)}$$

            $$B(α, β) = \int_0^1 t^{α - 1} \cdot (1 - t)^{β - 1} dt$$

            Where:
            - $$x$$: The random variable between 0 and 1.
            - $$α (alpah)$$: A parameter that controls the shape of the distribution on the left side. Higher values of α result in a distribution more concentrated towards the left (closer to 0).
            - $$β (beta)$$: A parameter that controls the shape of the distribution on the right side. Higher values of β result in a distribution more concentrated towards the right (closer to 1).
            - $$x^{α - 1} \cdot (1 - x)^{β - 1}$$: The numerator represents the probability density at a given point.
            - $$B(α, β)$$: A normalizing factor ensuring that the total area under the probability density function is 1.
            - $$t$$: The possible values of the random variable in the range of 0 to 1.
            - $$t^{α - 1}$$: Contributes to the probability density, with $$α - 1$$ providing the power to $$t$$.
            - $$t^{β - 1}$$: Contributes to the probability density, with $$β - 1$$ providing the power to $$1 - t$$.
            - $$\int_0^1$$: The definite integral, indicating that we are summing up the contributions of the probability density function over the entire range of possible values (from 0 to 1).
            """,
            mathjax=True
        )
    ]),
    html.H2(className='section-title', children='Examples'),
    html.Div(className='paragraph', children=[
        html.P(children=["1. To model the proportion of a portfolio invested in different assets."]),
        html.P(children=["2. To estimate the probability of default in credit risk modeling."]),
        html.P(children=["3. To predict the conversion rates of online advertisements or marketing campaigns."]),
        html.P(children=["4. To model the proportion of market share held by different products."]),
        html.P(children=["5. To assess the effectiveness of a drug treatment based on the proportion of positive outcomes."]),
        html.P(children=["6. To estimate the proportion of defective items in a manufacturing process."]),
        html.P(children=["7. To predict the proportion of students passing an exam."]),
        html.P(children=["8. To estimate the probability of a policyholder filing a claim."]),
    ])
])


@callback(
    Output('plot-pdf-beta', 'figure'),
    Output('plot-cdf-beta', 'figure'),
    Input('input-alpha-beta', 'value'),
    Input('input-beta-beta', 'value'),
)
def pdf_cdf_beta(
    alpha: float,
    beta_: float,
) -> Tuple[plotly_figure, plotly_figure]:
    """
    Plot the probability density function for a beta distribution
    and its cumulative distribution function.
    """

    x = np.linspace(0, 1, 200)

    y_pdf = beta.pdf(x, alpha, beta_)
    fig_pdf = go.Figure(
        data=go.Scatter(
            x=x,
            y=y_pdf,
            hovertemplate='x: %{x:.2f}<br>Probability Density: %{y:.2f}<extra></extra>'
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

    y_cdf = beta.cdf(x, alpha, beta_)
    fig_cdf = go.Figure(
        data=go.Scatter(
            x=x,
            y=y_cdf,
            hovertemplate='x: %{x:.2f}<br>Probability Density: %{y:.2f}<extra></extra>'
        )
    )
    fig_cdf.update_layout(
        title=dict(
            text='Cumulative Distribution Function',
            x=0.5,
        ),
        xaxis=dict(title='X'),
        yaxis=dict(title='Probability Density'),
    )

    return fig_pdf, fig_cdf
