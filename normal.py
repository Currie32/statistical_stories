from dash import dcc, html
import plotly.express as px
from dash.dependencies import Input, Output
from scipy.stats import norm
import numpy as np
import plotly.graph_objects as go


normal_layout = html.Div([
    html.H1(className='content-title', children='Normal Distribution'),
    html.Div(className='plot-parameters', children=[
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='Mean'),
            dcc.Input(id='normal-mean-input', value=0, min=-10, max=10, step=0.1, type='number'),
        ]),
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='Standard Deviation'),
            dcc.Input(id='normal-std-input', value=1, min=0, max=10, step=0.1, type='number'),
        ])
    ]),
    html.Div(className='plots-distribution', children=[
        html.Div(className='plot', children=[dcc.Graph(id='normal-plot')]),
        html.Div(className='plot', children=[dcc.Graph(id='normal-cdf-plot')])
    ])
])


def update_normal_plot(mean, std):
    x = np.linspace(-10, 10, 200)
    y_pdf = norm.pdf(x, loc=mean, scale=std)
    y_cdf = norm.cdf(x, loc=mean, scale=std)

    fig_pdf = go.Figure(data=go.Scatter(x=x, y=y_pdf))
    fig_pdf.update_layout(
        title='Probability Density Function',
    )

    fig_cdf = go.Figure(data=go.Scatter(x=x, y=y_cdf))
    fig_cdf.update_layout(title='Cumulative Density Function')

    return fig_pdf, fig_cdf
