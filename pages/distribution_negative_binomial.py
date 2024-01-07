from typing import Tuple

import numpy as np
import plotly.graph_objects as go
from dash import dcc, html, register_page, Input, Output, callback
from plotly.graph_objs._figure import Figure as plotly_figure
from scipy.stats import nbinom


register_page(__name__, path="/negative-binomial")

layout = html.Div(className='content', children=[
    html.H1(className='content-title', children='Negative Binomial Distribution'),
    html.Div(
        className="resource-link",
        children=[html.A("Link to scipy", target="_blank", href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nbinom.html")]
    ),
    html.H2(className='section-title', children='Overview'),
    html.Div(className='paragraph', children=[
        html.P("Once upon a time, there was a man named Dave who loved playing basketball. He had a goal of making as many three-point shots as possible in a game. One day, he decided to keep track of his shots and count how many attempts it took for him to make five shots."),
        html.P("As Dave started playing, he quickly realized that he didn't always make a shot on his first try. Sometimes it took him two attempts, sometimes three, and occasionally even more. Curious about this pattern, Dave decided to analyze his shooting performance using something called the negative binomial distribution."),
        html.P("Dave understood that the negative binomial distribution could help him predict how many attempts it would take for him to make a certain number of shots. In this case, his target was five made shots. The negative binomial distribution focuses on the number of trials or attempts needed until a specific number of successes occur."),
        html.P("So, Dave began collecting data during his basketball practices. He recorded the number of attempts it took him to make five shots each time. After several rounds, he had a list of results."),
        html.P("As he analyzed the data, Dave noticed that the probability of making a shot on each attempt remained constant. It was like flipping a coin with a fixed probability of heads or tails. Sometimes he made a shot, and sometimes he didn't."),
        html.P("To understand the negative binomial distribution better, Dave decided to visualize it. He created a chart where the x-axis represented the number of attempts required to make five shots, and the y-axis represented the probability of this requirement. The shape of the chart resembled a skewed curve."),
        html.P("When Dave examined the chart, he noticed a few interesting things. Firstly, the peak of the curve indicated the average number of attempts it took for him to make five shots. Secondly, the curve gradually declined as the number of attempts increased, showing that the probability of requiring many attempts was less likely."),
        html.P("Dave realized that the negative binomial distribution helped him understand the likelihood of achieving his goal based on his shooting performance. It gave him insights into the average number of attempts it took for him to make five shots, as well as the variation in his performance from game to game."),
        html.P("Armed with this knowledge, Dave could set realistic expectations for himself. He understood that if his shooting skills improved, he could expect to make five shots in fewer attempts. Conversely, if he was having an off day, he might need more attempts to reach his target."),
        html.P("As Dave continued playing basketball and tracking his shooting performance, he found the negative binomial distribution to be a useful tool in understanding his progress. It allowed him to set goals, monitor his improvement, and adjust his strategy accordingly."),
    ]),
    html.H2(className='section-title', children='Summary'),
    html.Div(className='paragraph', children=[
        html.P("The negative binomial distribution describes the number of independent and identically distributed Bernoulli trials needed for a predetermined number of failures to occur, with a given probability of success on each trial.")
    ]),
    html.H2(className='section-title', children='Visualizing the Negative Binomial Distribution'),
    html.Div(className='plot-parameters', children=[
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='Number of Successes'),
            dcc.Input(className='parameter-value', id='input-successes-negative-binomial', value=5, min=1, max=100, step=1, type='number'),
        ]),
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='Probability of Success'),
            dcc.Input(className='parameter-value', id='input-probability-negative-binomial', value=0.3, min=0, max=1, step=0.01, type='number'),
        ])
    ]),
    html.Div(className='plots-two', children=[
        html.Div(className='plot', children=[dcc.Graph(id='plot-pmf-negative-binomial')]),
        html.Div(className='plot', children=[dcc.Graph(id='plot-cdf-negative-binomial')])
    ]),
    html.H2(className='section-title', children='Assumptions'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("1. Independent and Identically Distributed (IID) Trials: "), "Each trial or attempt is independent of the others."]),
        html.P(children=[html.Strong("2. Fixed Probability of Success: "), "The probability of success (e.g., making a shot) remains constant in each trial."]),
        html.P(children=[html.Strong("3. Discrete Outcomes: "), "The outcome is a discrete (i.e. integer) variable, instead of a continuous variables."]),
    ]),
    html.H2(className='section-title', children='Formula'),
    html.Div(className='paragraph', children=[
        dcc.Markdown(
            r"""
            $$P(X = k) = \binom{r + k - 1}{k} * p^r * (1 - p)^k$$

            Where:
            - $$P(X = k)$$: The probability that the random variable $$X$$ takes on the value $$k$$.
            - $$r$$: The number of successes needed before the experiment is complete.
            - $$k$$: The number of failures until the required number of successes is reached.
            - $$p$$: The probability of success in each trial.
            - $$(1 - p)$$: The probability of failure in each trial.
            - $$\binom{r + k - 1}{k}$$: The binomial coefficient, also known as "n choose k." It calculates the number of ways to choose $$k$$ failures out of $$(r + k - 1)$$ trials. It is calculated as $$\binom{r + k - 1}{k}$$ = $$\frac{(r + k - 1)!}{(r - 1)! (k)!}$$, where $$!$$ denotes the factorial.
            """,
            mathjax=True
        )
    ]),
    html.H2(className='section-title', children='Examples'),
    html.Div(className='paragraph', children=[
        html.P(children=["1. To model the number of defects or faulty units in a production process before a specified number of acceptable units is produced."]),
        html.P(children=["2. To analyze the number of customer complaints or service requests received before a certain number of satisfactory resolutions is achieved."]),
        html.P(children=["3. To estimate the number of claims an insurance company receives before a predetermined number of high-cost claims occur."]),
        html.P(children=["4. To model the number of disease outbreaks or infections until a certain number of hospitalizations is reached."]),
        html.P(children=["5. To model the number of website visits before a certain number of purchases occur."]),
    ]),
])


@callback(
    Output('plot-pmf-negative-binomial', 'figure'),
    Output('plot-cdf-negative-binomial', 'figure'),
    Input('input-successes-negative-binomial', 'value'),
    Input('input-probability-negative-binomial', 'value'),
)
def pmf_cdf_negative_binomial(
    n_successes: int,
    prob_success: float,
) -> Tuple[plotly_figure, plotly_figure]:
    """
    Plot the probability mass function for a negative binomial distribution
    and its cumulative distribution function.
    """

    x = list(range(0, 50))

    y_pmf = nbinom.pmf(x, n_successes, prob_success)
    fig_pmf = go.Figure(
        data=go.Bar(
            x=x,
            y=y_pmf,
            hovertemplate='# failures: %{x}<br>Probability: %{y:.2f}<extra></extra>'
        )
    )
    fig_pmf.update_layout(
        title=dict(
            text='Probability Mass Function',
            x=0.5,
        ),
        xaxis=dict(title='Number of Failures'),
        yaxis=dict(title='Probability'),
    )

    y_cdf = nbinom.cdf(x, n_successes, prob_success)
    fig_cdf = go.Figure(
        data=go.Bar(
            x=x,
            y=y_cdf,
            hovertemplate='# failures: %{x}<br>Probability: %{y:.2f}<extra></extra>'
        )
    )
    fig_cdf.update_layout(
        title=dict(
            text='Cumulative Distribution Function',
            x=0.5,
        ),
        xaxis=dict(title='Number of Failures'),
        yaxis=dict(title='Probability'),
    )

    return fig_pmf, fig_cdf
