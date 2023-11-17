from typing import Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
from dash import callback, dcc, html, Input, Output, register_page
from plotly.graph_objs._figure import Figure as plotly_figure
from statsmodels.formula.api import ols


register_page(__name__, path="/anova")

layout = html.Div(className='content', children=[
    html.H1(className='content-title', children='ANOVA (ANalysis Of VAriance)'),
    html.Div(
        className="resource-link",
        children=[html.A("Link to statsmodels", target="_blank", href="https://www.statsmodels.org/stable/anova.html")]
    ),
    html.H2(className='section-title', children='Overview'),
    html.Div(className='paragraph', children=[
        html.P("Once upon a time, there was an orchard owner named Dave. His orchard was filled with three types of fruit-bearing trees: Apple, Orange, and Mango. Dave loved experimenting with his trees and was always curious to see if there were any significant differences between their yields."),
        html.P("One sunny day, Dave decided to explore a statistical tool called ANOVA, short for Analysis of Variance, to help him understand the differences in yield among his orchard trees."),
        html.P("To begin his investigation, Dave carefully selected ten random apple trees, ten orange trees, and ten mango trees. He carefully noted down all the data and organized it in a table."),
        html.P("To apply ANOVA to his data, Dave needed to calculate the average yield for each tree type. After calculating this, Dave discovered that the Apple trees had an average yield of 200 fruits, the Orange trees had an average yield of 180 fruits, and the Mango trees had an average yield of 220 fruits."),
        html.P("Curiosity piqued, Dave wondered if these differences were statistically significant or simply due to random chance. He knew that ANOVA could help him find out."),
        html.P("Dave learned that ANOVA compares two types of variability: the variability within each group and the variability between the groups."),
        html.P("Within-group variability measures how much the fruit yields vary within each type of tree. If there is a lot of variation within each group, it suggests that the fruit yields are not very similar for that type of tree."),
        html.P("Between-group variability, on the other hand, measures how much the average fruit yields differ between the three types of trees. If there is a large difference between the average yields of the different types of trees, it suggests that the groups themselves are different."),
        html.P("Dave knew that ANOVA could calculate these variabilities using a special formula that takes into account the number of trees in each group, the means, and the individual fruit yield measurements. By plugging in his data, he could obtain two crucial values: the between-group sum of squares (SSB) and the within-group sum of squares (SSW). The ratio of these two values creates a statistical measure called the F statistic, which determines if the results are significant."),
        html.P("If the SSB is significantly larger than the SSW, it indicates that there is a strong likelihood that the average fruit yields of the different types of trees are indeed different. Conversely, if the SSW is larger, it suggests that the differences between the groups are likely due to random chance, and the average fruit yields are not significantly different."),
        html.P("Excited to unveil the truth, Dave eagerly applied ANOVA to his orchard data. The results showed that the SSB was substantially larger than the SSW. This meant that there was a significant difference between the three groups of trees. In simpler terms, the average fruit yields of Apple, Orange, and Mango trees were not the same."),
        html.P("Delighted by his discovery, Dave concluded that ANOVA was a powerful tool for determining if there were any meaningful differences in fruit yields between different types of trees. It enabled him to make informed decisions based on the data he collected, guiding him to optimize his orchard's productivity."),
        html.P("From that day forward, Dave continued to explore the world of statistics and used ANOVA to analyze various aspects of his orchard. He was always eager to uncover hidden insights and make data-driven choices to nurture his trees and achieve bountiful harvests year after year."),
        ]),
    html.H2(className='section-title', children='Visualizing ANOVA'),
    html.Button('Generate New Data', id='button-new-data', n_clicks=0),
    html.Div(className='plot-parameters', children=[
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='Mean (A)'),
            dcc.Input(className='parameter-value', id='input-mean-a-anova', value=20, min=0, max=100, step=1, type='number'),
            html.Label(className='parameter-label', children='Std Dev (A)'),
            dcc.Input(className='parameter-value', id='input-std-a-anova', value=6, min=1, max=100, step=1, type='number'),
            html.Label(className='parameter-label', children='Samples (A)'),
            dcc.Input(className='parameter-value', id='input-samples-a-anova', value=28, min=1, max=100, step=1, type='number'),
        ]),
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='Mean (B)'),
            dcc.Input(className='parameter-value', id='input-mean-b-anova', value=22, min=0, max=100, step=1, type='number'),
            html.Label(className='parameter-label', children='Std Dev (B)'),
            dcc.Input(className='parameter-value', id='input-std-b-anova', value=7, min=1, max=100, step=1, type='number'),
            html.Label(className='parameter-label', children='Samples (B)'),
            dcc.Input(className='parameter-value', id='input-samples-b-anova', value=10, min=1, max=100, step=1, type='number'),
        ]),
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='Mean (C)'),
            dcc.Input(className='parameter-value', id='input-mean-c-anova', value=18, min=0, max=100, step=1, type='number'),
            html.Label(className='parameter-label', children='Std Dev (C)'),
            dcc.Input(className='parameter-value', id='input-std-c-anova', value=5, min=1, max=100, step=1, type='number'),
            html.Label(className='parameter-label', children='Samples (C)'),
            dcc.Input(className='parameter-value', id='input-samples-c-anova', value=18, min=1, max=100, step=1, type='number'),
        ]),
    ]),
    html.Div(className='table-full-width', children=[html.Table(id='table-anova')]),
    html.Div(className='plot-full-width', children=[dcc.Graph(id='plot-anova')]),
    html.H2(className='section-title', children='Assumptions'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("1. Independence: "), "Measurements or data points collected from one group should not be influenced by or dependent on the measurements from another group."]),
        html.P(children=[
            html.Strong("2. Normality: "),
            "The data within each group should follow a normal distribution. Normality can be assessed through graphical methods like histograms or quantile-quantile (Q-Q) plots, or through statistical tests such as the ",
            html.A("Shapiro-Wilk test", target="_blank", href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html"),
            ".",
        ]),
        html.P(children=[
            html.Strong("3. Homogeneity of Variance: "),
            "The variability, or spread, of data within each group should be roughly equal across all groups. This assumption can be evaluated by examining the variability within each group or formally tested using statistical tests like ",
            html.A("Levene's test ", target="_blank", href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html"),
            "or ",
            html.A("Bartlett's test", target="_blank", href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bartlett.html"),
            ".",
        ]),
    ]),
    html.H2(className='section-title', children='Alternative tests'),
    html.Div(className='paragraph', children=[
        html.P(children=[
            html.Strong(
                html.A("Kruskal-Wallis Test:", target="_blank", href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html"),
            ),
            " A non-parametric alternative when the assumption of normality is violated. It compares the medians of two or more groups by ranking the data and testing if the distribution of ranks differ significantly among the groups."
        ]),
        html.P(children=[
            html.Strong(
                html.A("Friedman Test:", target="_blank", href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.friedmanchisquare.html"),
            ),
            " A non-parametric alternative when the assumption of normality is violated. It comparesd three or more groups by ranking the data within each group and testing if the rankings differ significantly among the groups."
        ]),
        html.P(children=[
            html.Strong(
                html.A("Brown-Forsythe Test:", target="_blank", href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html"),
            ),
            " Can be used when the assumption of homogeneity of variance is violated. It works by measuring the spread in each group by performing an ANOVA on a transformation of the response variable and is more robust to unequal sample sizes than Welch's F-test."
        ]),        
    ]),
    html.H2(className='section-title', children='Formula'),
    html.Div(className='paragraph', children=[
        dcc.Markdown(
            r"""
            $$F = \frac{SSB / (k - 1)}{SSW / (n - k)}$$

            $$SSB = \sum_{i=1}^{k} n_i(\bar{X_i} - \bar{X})^2$$
            
            $$SSW = \sum_{i=1}^{k} \sum_{j=1}^{n_i} (X_{ij} - \bar{X_i})^2$$


            Where:
            - $$F$$: The F-statistic is the test statistic used in ANOVA. It represents the ratio of the between-group variability to the within-group variability. By comparing the F-statistic to the critical value from the F-distribution, we can determine if the differences between the group means are statistically significant.
            - $$SSB$$: The Sum of Squares Between groups represents the variability or differences between the group means. It measures how much the group means deviate from the overall mean.
            - $$SSW$$: The Sum of Squares Within groups represents the variability or differences within each group. It measures how much the individual observations deviate from their respective group means.
            - $$k$$: The number of groups (or treatment levels). $$k - 1$$ is the degrees of freedom for SSB.
            - $$n$$: The total number of observations. $$n - k$$ is the degrees of freedom for SSW.
            - $$n_i$$: The number of observations in group i.
            - $$\bar{X_i}$$: The mean of group i.
            - $$\bar{X}$$: The overall mean across all groups.
            - $$X_{ij}$$: The j^th observation in group i.
            """,
            mathjax=True
        )
    ]),
    html.H2(className='section-title', children='Example use cases'),
    html.Div(className='paragraph', children=[
        html.P(children=["1. Compare the effects of different fertilizers, pesticides, or irrigation methods on crop yields."]),
        html.P(children=["2. Compare the effectiveness of different teaching methods, curriculum approaches, or interventions on student learning outcomes"]),
        html.P(children=["3. Compare the preferences or perceptions of different consumer groups towards products or advertisements."]),
        html.P(children=["4. Compare the effects of different treatments or interventions on patient outcomes."]),
        html.P(children=["5. Compare the impact of different factors (such as pollution levels, temperature variations, or habitat types) on biodiversity or ecological parameters"]),
    ]),
])


@callback(
    Output('table-anova', 'children'),
    Output('plot-anova', 'figure'),
    Input('button-new-data', 'n_clicks'),
    Input('input-mean-a-anova', 'value'),
    Input('input-std-a-anova', 'value'),
    Input('input-samples-a-anova', 'value'),
    Input('input-mean-b-anova', 'value'),
    Input('input-std-b-anova', 'value'),
    Input('input-samples-b-anova', 'value'),
    Input('input-mean-c-anova', 'value'),
    Input('input-std-c-anova', 'value'),
    Input('input-samples-c-anova', 'value'),
)
def anova(
    n_clicks: int, # Not used, but a required input to generate new data
    mean_a: int, std_dev_a: int, samples_a: int,
    mean_b: int, std_dev_b: int, samples_b: int,
    mean_c: int, std_dev_c: int, samples_c: int,
) -> Tuple[html.Table, plotly_figure]:
    """
    Generate three samples from normal distributions.
    Perform ANOVA on the three samples.
    Output the results in a table and boxplots.
    """    
    # Generate data using normal distribution
    np.random.seed(n_clicks)
    data_a = pd.DataFrame({'Group': ['A'] * samples_a, 'Value': np.random.normal(mean_a, std_dev_a, samples_a)})
    np.random.seed(n_clicks)
    data_b = pd.DataFrame({'Group': ['B'] * samples_b, 'Value': np.random.normal(mean_b, std_dev_b, samples_b)})
    np.random.seed(n_clicks)
    data_c = pd.DataFrame({'Group': ['C'] * samples_c, 'Value': np.random.normal(mean_c, std_dev_c, samples_c)})

    # Combine the data
    df = pd.concat([data_a, data_b, data_c], ignore_index=True)
    # Round values to two decimals places. This makes the tooltip look nicer
    df['Value'] = df['Value'].round(2)
    
    # Perform ANOVA
    model = ols('Value ~ Group', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    table = html.Table(
        id='table-anova',
        children=[
            html.Thead(html.Tr([
                html.Th("Source of Variation"),
                html.Th("Sum of Squares"),
                html.Th("Degrees of Freedom"),
                html.Th("F-Value"),
                html.Th("p-value"),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(anova_table.index[i]),
                    html.Td(round(anova_table['sum_sq'][i], 2)),
                    html.Td(round(anova_table['df'][i], 2)),
                    html.Td(round(anova_table['F'][i], 3)),
                    html.Td(round(anova_table['PR(>F)'][i], 3))
                ]) for i in range(len(anova_table))
            ])
        ]
    )

    fig = px.box(df, x='Group', y='Value', points='all')
    fig.update_layout()

    return table, fig
