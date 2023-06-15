from dash import dcc, html
import plotly.express as px
from dash.dependencies import Input, Output
from scipy.stats import poisson
import numpy as np
import plotly.graph_objects as go


poisson_layout = html.Div([
    html.H1('Poisson Distribution'),
    html.Div(className='row', children=[
        html.Div(className='col-md-6', children=[
            html.Label('Lambda'),
            dcc.Input(
                id='poisson-lambda-input',
                type='number',
                value=1,
                step=0.1,
                min=0,
                max=10
            ),
            dcc.Graph(id='poisson-plot')
        ]),
        html.Div(className='col-md-6', children=[
            dcc.Graph(id='poisson-cdf-plot')
        ])
    ])
])


def update_poisson_plot(lam):
    x = np.arange(0, 21)
    y_pdf = poisson.pmf(x, lam)
    y_cdf = poisson.cdf(x, lam)

    fig_pdf = px.bar(x=x, y=y_pdf)
    fig_pdf.update_layout(title='Poisson Distribution - Probability Mass Function')

    fig_cdf = go.Figure(data=go.Scatter(x=x, y=y_cdf))
    fig_cdf.update_layout(title='Poisson Distribution - Cumulative Mass Function')

    return fig_pdf, fig_cdf
