from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import callback, dcc, html, Input, Output, register_page
from plotly.graph_objs._figure import Figure as plotly_figure
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency


register_page(__name__, path="/chi-square")

layout = html.Div(className='content', children=[
    html.H1(className='content-title', children='Chi-Square Test'),
    html.Div(
        className="resource-link",
        children=[html.A("Link to scipy", target="_blank", href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html")]
    ),
    html.H2(className='section-title', children='Overview'),
    html.Div(className='paragraph', children=[
        html.P("Once upon a time, there was an enthusiastic gardener named Dave. He loved spending time in his backyard, nurturing various plants and exploring the wonders of nature. One day, while strolling through a local farmer's market, Dave came across three different types of tomato plants: red, yellow, and green. Excited by the possibilities, he decided to grow these tomatoes in his garden and see how they would thrive."),
        html.P("Dave divided his garden into two sections, labelling them 'Section A' and 'Section B.' In Section A, he used traditional soil, and in Section B, he experimented with a new nutrient-rich soil mix. With great care, Dave planted an equal number of tomato plants from each colour group in both sections."),
        html.P("As the weeks went by, Dave diligently watered and tended to his tomato plants, eagerly awaiting the day they would bear fruit. Finally, the moment arrived. Dave marvelled at the sight of his garden bursting with vibrant red, yellow, and green tomatoes. But he began to wonder if the type of soil had influenced the distribution of tomato colours across the sections."),
        html.P("Curiosity getting the better of him, Dave decided to investigate further. He turned to the power of statistics to help him uncover any potential associations. With a little research, he discovered a statistical test called the chi-square test, which could determine whether there was a significant relationship between two categorical variables."),
        html.P("Armed with newfound knowledge, Dave set out to conduct the chi-square test on his tomato data. He started by counting the number of red, yellow, and green tomatoes in each section. In Section A, he found 30 red, 20 yellow, and 10 green tomatoes. In Section B, the counts were 25 red, 15 yellow, and 20 green tomatoes."),
        html.P("Dave began constructing a contingency table, using the columns to represent the tomato colours (red, yellow, and green) and the rows to represent the sections (X and Y). He filled in the observed values accordingly."),
        html.P("To proceed with the chi-square test, Dave needed to calculate the expected values for each cell in the contingency table. The expected values represent what he would expect to see if there were no relationship between the type of soil and tomato colour. Utilizing the formula:"),
        html.P("Expected value = (sum of row total * sum of column total) / total number of observations"),
        html.P("Dave calculated the expected values and filled them into the contingency table. Now, it was time to compute the chi-square test statistic."),
        html.P("Applying the chi-square formula, Dave found that the test statistic was 4.5. The next step was to compare this value with the critical value from a chi-square distribution table. The critical value is determined based on the degrees of freedom, which depend on the dimensions of the contingency table."),
        html.P("After examining the table, Dave concluded that 4.5 was less than the critical value. This indicated that there was not a significant association between the type of soil and tomato colour."),
        html.P("Although Dave didn't find a significant relationship, he was not disheartened. He realized that the chi-square test had provided valuable insights into the distribution of tomato colours across the two sections of his garden. It had helped him analyze the data and make informed conclusions."),
    ]),
    html.H2(className='section-title', children='Visualizing the Chi-Square Test'),
    html.Div(className='plot-parameters', children=[
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='Red (A)'),
            dcc.Input(className='parameter-value', id='input-a2-chi-square', value=30, min=1, max=100, step=1, type='number'),
            html.Label(className='parameter-label', children='Red (B)'),
            dcc.Input(className='parameter-value', id='input-a1-chi-square', value=25, min=1, max=100, step=1, type='number'),
        ]),
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='Yellow (A)'),
            dcc.Input(className='parameter-value', id='input-b2-chi-square', value=20, min=1, max=100, step=1, type='number'),
            html.Label(className='parameter-label', children='Yellow (B)'),
            dcc.Input(className='parameter-value', id='input-b1-chi-square', value=15, min=1, max=100, step=1, type='number'),
        ]),
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='Green (A)'),
            dcc.Input(className='parameter-value', id='input-c2-chi-square', value=10, min=1, max=100, step=1, type='number'),
            html.Label(className='parameter-label', children='Green (B)'),
            dcc.Input(className='parameter-value', id='input-c1-chi-square', value=20, min=1, max=100, step=1, type='number'),
        ]),
    ]),
    html.P(className='statistical-result', id='result-chi-square'),
    html.Div(className='plot-full-width', children=[dcc.Graph(id='plot-chi-square')]),
    html.H2(className='section-title', children='Assumptions'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("1. Independence: "), "The occurrence of one observation should not influence or be influenced by another observation."]),
        html.P(children=[html.Strong("2. Categorical: "), "Both variables are for categorical data."]),
        html.P(children=[html.Strong("3. Mutually Exclusive: "), "Observations can only belong to one cell in the contingency table."]),
        html.P(children=[html.Strong("4. Sample Size: "), "There should be at least five observations in each cell of the contingency table."]),

    ]),
    html.H2(className='section-title', children='Alternative tests'),
    html.Div(className='paragraph', children=[
        html.P(children=[
            html.Strong(
                html.A("Fisher's Exact Test:", target="_blank", href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html"),
            ),
            " Suitable when the sample size is small, and the expected cell frequencies in the contingency table are less than 5. It is often used as an alternative to the chi-square test in 2x2 contingency tables."
        ]),
        html.P(children=[
            html.Strong(
                html.A("McNemar's Test:", target="_blank", href="https://www.statsmodels.org/dev/generated/statsmodels.stats.contingency_tables.mcnemar.html"),
            ),
            " Used when analyzing paired categorical data, typically in a 2x2 contingency table, where the observations are dependent or related. It is commonly used in before-after studies or matched case-control studies."
        ]),
        html.P(children=[
            html.Strong(
                html.A("Cochran-Mantel-Haenszel Test:", target="_blank", href="https://www.statsmodels.org/dev/generated/statsmodels.stats.contingency_tables.StratifiedTable.html"),
            ),
            " Used when analyzing categorical data in stratified or matched studies. It allows for the comparison of multiple 2x2 contingency tables while controlling for confounding variables or stratification factors."
        ]),
    ]),
    html.H2(className='section-title', children='Formula'),
    html.Div(className='paragraph', children=[
        dcc.Markdown(
            r"""
            $$\chi^2 = \sum \frac {(O_{ij} - E_{ij})^2}{E_{ij}}$$

            Where:
            - $$\chi^2$$: The chi-square test statistic, which measures the discrepancy between the observed and expected values.
            - $$Σ$$ $$(sigma)$$: Sum the values for each cell of the contingency table.
            - $$O_{ij}$$: The observed frequency in each cell of the contingency table.
            - $$E_{ij}$$: The expected frequency in each cell of the contingency table.
            """,
            mathjax=True
        )
    ]),
    html.H2(className='section-title', children='Example use cases'),
    html.Div(className='paragraph', children=[
        html.P(children=["1. Assess the relationship between demographic variables (e.g., age, gender, income) and consumer preferences or buying behavior."]),
        html.P(children=["2. Examine the association between risk factors and disease outcomes, such as smoking status and lung cancer incidence."]),
        html.P(children=["3. Explore the relationship between categorical variables such as educational attainment and employment status or political affiliation and voting behavior."]),
        html.P(children=["4. Determine if observed patterns of inheritance are consistent with the expected Mendelian ratios, or if certain genetic markers are associated with specific traits or diseases."]),
        html.P(children=["5. Assess the relationship between quality control variables, such as the type of defect, and the production line."]),
    ]),
])


@callback(
    Output('plot-chi-square', 'figure'),
    Output('result-chi-square', 'children'),
    Input('input-a1-chi-square', 'value'),
    Input('input-a2-chi-square', 'value'),
    Input('input-b1-chi-square', 'value'),
    Input('input-b2-chi-square', 'value'),
    Input('input-c1-chi-square', 'value'),
    Input('input-c2-chi-square', 'value'),
)
def chi_square(
    a1: int, a2: int,
    b1: int, b2: int,
    c1: int, c2: int,
) -> Tuple[plotly_figure, plotly_figure]:
    """
    Generate three samples from normal distributions.
    Perform ANOVA on the three samples.
    Output the results in a table and boxplots.
    """
    # Create the observed data
    observed_data = np.array([[a1, b1, c1], [a2, b2, c2]])

    # Compute the expected values
    row_totals = observed_data.sum(axis=1)
    col_totals = observed_data.sum(axis=0)
    total_observations = observed_data.sum()
    expected_data = np.outer(row_totals, col_totals) / total_observations

    # Create a contingency table using pandas DataFrame
    columns = ['Red', 'Yellow', 'Green']
    rows = ['B', 'A']

    # Create the subplots for observed and expected heatmaps
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Observed Data', 'Expected Data'))

    # Add the observed heatmap
    fig.add_trace(
        go.Heatmap(
            x=columns,
            y=rows,
            z=observed_data,
            colorscale="blues",
            colorbar=dict(title='Counts'),
            hovertemplate="Section: %{y}<br>Colour: %{x}<br>Count: %{z}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add the expected heatmap
    fig.add_trace(
        go.Heatmap(
            x=columns,
            y=rows,
            z=expected_data,
            colorscale="blues",
            colorbar=dict(title='Counts'),
            hovertemplate="Section: %{y}<br>Colour: %{x}<br>Count: %{z}<extra></extra>",
            showscale=False
        ),
        row=1,
        col=2,
    )

    # Set the layout
    fig.update_layout(
        title='Observed vs Expected Data',
        xaxis=dict(title='Colour'),
        yaxis=dict(title='Section'),
    )

    # Perform the chi-square test
    chi_sq, p_value, _, _ = chi2_contingency(observed_data, correction=False)
    chi_sq = round(chi_sq, 3)
    p_value = round(p_value, 3)
    text = f"Test statistic: {chi_sq}, p-value: {p_value}"

    return fig, text
