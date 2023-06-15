from dash import dcc, html
import dash
import plotly.express as px
from dash.dependencies import Input, Output
from scipy.stats import norm
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro


normal_layout = html.Div(className='content', children=[
    html.H1(className='content-title', children='Normal Distribution'),
    html.H2(className='section-title', children='Overview'),
    html.Div(className='paragraph', children=[
        html.P("Once upon a time in a small village, there was a clever farmer named John. John had a big field where he grew his crops, and he was always curious about how his crops grew. He wanted to understand how the height of his plants varied from one another."),
        html.P("One day, John decided to measure the height of all the cornstalks in his field. He found that most of the cornstalks were of average height, neither too tall nor too short. However, there were a few cornstalks that were exceptionally tall, and a few that were very short."),
        html.P("Intrigued by this observation, John decided to plot a graph to represent the heights of his cornstalks. He made a horizontal line and marked different heights of the cornstalks along it. On the vertical axis, he marked the number of cornstalks with each height."),
        html.P("As John plotted the graph, he noticed something interesting. The graph had a bell-shaped curve! Most of the cornstalks were in the middle, around the average height, forming the highest point of the curve. As he moved away from the average height, the number of cornstalks decreased, forming the sloping sides of the curve."),
        html.P("John was fascinated by this bell-shaped curve, and he learned that it was called the normal distribution (also known as the Gaussian Distribution). He discovered that the normal distribution is a mathematical pattern that often appears in nature and human-made systems."),
        html.P("He realized that the normal distribution can be seen in many things around us, not just the heights of cornstalks. For example, the distribution of people's heights, the test scores of students in a class, or even the weights of apples in a basket."),
        html.P("John also learned that the normal distribution has some unique properties. He found out that the average height of the cornstalks was right in the middle of the curve, called the mean. He also learned that the curve was symmetrical, meaning that the left and right sides were mirror images of each other."),
        html.P("Furthermore, John discovered that the normal distribution had a standard deviation. The standard deviation is a measure of how spread out the data is. If the standard deviation is small, the data points are closely packed together, creating a narrow curve. If the standard deviation is large, the data points are more spread out, making the curve wider."),
        html.P("With this newfound knowledge, John was able to make predictions about his cornfield. He could estimate how many cornstalks would fall within a certain height range based on the properties of the normal distribution."),
    ]),
    html.H2(className='section-title', children='Visualizing the Normal Distribution'),
    html.Div(className='plot-parameters', children=[
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='Mean'),
            dcc.Input(id='normal-mean-input', value=0, min=-1000, max=1000, step=0.1, type='number'),
        ]),
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='Standard Deviation'),
            dcc.Input(id='normal-std-input', value=1, min=-1000, max=1000, step=0.1, type='number'),
        ])
    ]),
    html.Div(className='plots-distribution', children=[
        html.Div(className='plot', children=[dcc.Graph(id='normal-plot')]),
        html.Div(className='plot', children=[dcc.Graph(id='normal-cdf-plot')])
    ]),
    html.H2(className='section-title', children='Assumptions'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("1. Independence: "), "The observations or data points being analyzed should be independent of each other. In other words, one observation should not be influenced or dependent on another observation. This assumption is crucial for accurately estimating the parameters of the normal distribution."]),
        html.P(children=[html.Strong("2. Continuous Data: "), "The normal distribution assumes that the data being analyzed is continuous, meaning it can take any value within a range. Discrete or categorical data may not follow a normal distribution, and alternative distribution models may be more appropriate."]),
        html.P(children=[html.Strong("3. Symmetry: "), "The normal distribution assumes that the data is symmetrically distributed around its mean. This means that the left and right sides of the distribution are mirror images of each other. If the data is skewed (asymmetric), the normal distribution may not accurately represent the underlying distribution."]),
        html.P(children=[html.Strong("4: Constant Variance: "), "The variance of the data, which measures how spread out the observations are, is assumed to be constant across the entire range of values. This is known as homoscedasticity. If the variance is not constant (heteroscedasticity), the normal distribution may not be appropriate, and alternative models may be necessary."]),
        html.P(children=[html.Strong("5: Normality of Residuals: "), "In statistical modeling, the assumption is often made that the residuals (the differences between the observed values and the predicted values) follow a normal distribution. This assumption is crucial for certain statistical tests and inference procedures."]),
        html.P(children=[html.Strong("6: Outliers: "), "The normal distribution assumes that the data is free from outliers, which are extreme values that are far from the bulk of the data. Outliers can significantly affect the mean and standard deviation, potentially distorting the normal distribution. It is important to identify and handle outliers appropriately."]),
    ]),
    html.H2(className='section-title', children='Formula'),
    html.Div(className='paragraph', children=[
        dcc.Markdown(
            r"""
            $$f(x) = \frac{1}{{\sigma \sqrt{{2\pi}}}} \exp\left(-\frac{{(x - \mu)^2}}{{2\sigma^2}}\right)$$

            Where:
            - $$f(x)$$: This represents the probability density function, which gives the relative likelihood of observing a specific value (x) in the distribution.
            - $x$: This is the random variable, representing a particular value within the distribution for which we want to calculate the probability.
            - $μ$ $(mu)$: This is the mean of the distribution. It determines the center or average value around which the data is symmetrically distributed.
            - $σ$ $(sigma)$: This is the standard deviation of the distribution. It quantifies the spread or dispersion of the data points around the mean. A larger value of σ indicates a wider spread, while a smaller value indicates a narrower spread.
            - $exp()$: This is the exponential function, which raises the mathematical constant e (approximately 2.71828) to the power of the value within the parentheses.
            - $π$ $(pi)$: This is a mathematical constant approximately equal to 3.14159 and represents the ratio of the circumference of a circle to its diameter.
            - $√(2π)$: This represents the square root of 2π, which is a normalization factor that ensures the total area under the curve sums to 1. It standardizes the probability density function.
            """,
            mathjax=True
        )
    ]),
    html.H2(className='section-title', children='Examples'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("1. Heights and Weights: "), "Human heights and weights often follow a normal distribution. In fields like anthropology, clothing manufacturing, or nutrition, understanding the normal distribution of heights and weights helps design products that cater to the majority of the population."]),
        html.P(children=[html.Strong("2. Test Scores: "), "In education, the scores of large groups of students on standardized tests often approximate a normal distribution. This helps educators understand the performance of students and set appropriate grading criteria."]),
        html.P(children=[html.Strong("3. Financial Markets: "), "In finance and investing, the returns of many stocks or assets over a specific period often exhibit a normal distribution. This allows investors to estimate the probability of various outcomes and make informed decisions about their investments."]),
        html.P(children=[html.Strong("4. Quality Control: "), "In manufacturing and quality control processes, certain characteristics of products, such as weight, length, or volume, often follow a normal distribution. This helps determine acceptable ranges and identify potential defects."]),
        html.P(children=[html.Strong("5. Customer Satisfaction: "), "In market research and customer satisfaction surveys, the ratings provided by a large number of customers often approximate a normal distribution. This helps businesses gauge customer sentiment and identify areas for improvement."]),
        html.P(children=[html.Strong("6. Error Measurement: "), "In scientific experiments or engineering, measurement errors often exhibit a normal distribution. This allows researchers to estimate the precision and accuracy of their measurements and make statistically valid conclusions."]),
        html.P(children=[html.Strong("7. Time and Duration: "), "Various phenomena related to time, such as the duration of phone calls, waiting times, or service times, can often be modeled using a normal distribution. This helps in scheduling, resource allocation, and predicting service levels."]),
    ]),
    html.H2(className='section-title', children='Code'),
    html.Div(className='paragraph', children=[
        html.P(children=["This plot can be generated using the code below it."]),
    ]),
    html.Button('Generate New Data', id='button-new-data', n_clicks=0),
    dcc.Graph(id='histogram-plot'),
    html.H3(id='text-test-result'),
    html.Div(className='paragraph', children=[
        html.Pre(
            '''
# Set the mean and standard deviation of the normal distribution
mu = 0  # Mean
sigma = 1  # Standard deviation

# Generate a sample of 1000 data points from the normal distribution
sample = np.random.normal(mu, sigma, 1000)

# Create a histogram trace
histogram_trace = go.Histogram(
    x=sample,
    nbinsx=30,
    histnorm='probability density',
    opacity=0.7,
    marker=dict(color='lightblue', line=dict(color='darkblue', width=1)),
    showlegend=False
)

# Create a PDF trace
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
pdf = norm.pdf(x, mu, sigma)
pdf_trace = go.Scatter(
    x=x,
    y=pdf,
    mode='lines',
    line=dict(color='red', width=2),
    name='PDF'
)

# Create layout
layout = go.Layout(
    title='Normal Distribution',
    xaxis=dict(title='Value'),
    yaxis=dict(title='Probability Density'),
    showlegend=True
)

# Create figure
fig = go.Figure(data=[histogram_trace, pdf_trace], layout=layout)

# Perform Shapiro-Wilk normality test
_, p_value = shapiro(sample)
p_value = round(p_value, 3)

# Print the results
print(f'Statistic: {statistic:.4f}')
print(f'p-value: {p_value:.4f}')
if p_value > 0.05:
    print(f'The sample is likely drawn from a normal distribution (p-value = {p_value}).')
else:
    print(f'The sample does not appear to be from a normal distribution (p-value = {p_value}).')
            ''')
    ])
])


def update_normal_plot(mean, std):
    x = np.linspace(-10, 10, 200)
    y_pdf = norm.pdf(x, loc=mean, scale=std)
    y_cdf = norm.cdf(x, loc=mean, scale=std)

    fig_pdf = go.Figure(data=go.Scatter(x=x, y=y_pdf))
    fig_pdf.update_layout(
        title=dict(
            text='Probability Density Function',
            x=0.5,
        )
    )

    fig_cdf = go.Figure(data=go.Scatter(x=x, y=y_cdf))
    fig_cdf.update_layout(
        title=dict(
            text='Cumulative Density Function',
            x=0.5,
        )
    )

    return fig_pdf, fig_cdf


def plot_normal_distribution_with_pdf():
    # Set the mean and standard deviation of the normal distribution
    mu = 0  # Mean
    sigma = 1  # Standard deviation

    # Generate a sample of 1000 data points from the normal distribution
    sample = np.random.normal(mu, sigma, 1000)

    # Create a histogram trace
    histogram_trace = go.Histogram(
        x=sample,
        nbinsx=30,
        histnorm='probability density',
        opacity=0.7,
        marker=dict(color='lightblue', line=dict(color='darkblue', width=1)),
        showlegend=False
    )

    # Create a PDF trace
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
    pdf = norm.pdf(x, mu, sigma)
    pdf_trace = go.Scatter(
        x=x,
        y=pdf,
        mode='lines',
        line=dict(color='red', width=2),
        name='PDF'
    )

    # Create layout
    layout = go.Layout(
        title='Normal Distribution',
        xaxis=dict(title='Value'),
        yaxis=dict(title='Probability Density'),
        showlegend=True
    )

    # Create figure
    fig = go.Figure(data=[histogram_trace, pdf_trace], layout=layout)

    # Perform Shapiro-Wilk normality test
    _, p_value = shapiro(sample)
    p_value = round(p_value, 3)

    # Determine the normality test message
    if p_value > 0.05:
        normality_test_message = f'The sample is likely drawn from a normal distribution (p-value = {p_value}).'
    else:
        normality_test_message = f'The sample does not appear to be from a normal distribution (p-value = {p_value}).'

    # Return the figure and normality test message
    return fig, normality_test_message


