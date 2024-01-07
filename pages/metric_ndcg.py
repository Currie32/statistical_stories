from typing import Dict, List, Tuple

import numpy as np
from dash import callback, dcc, html, Input, Output, register_page
from dash import dash_table
from sklearn.metrics import ndcg_score


register_page(__name__, path="/ndcg")


layout = html.Div(className='content', children=[
    html.H1(className='content-title', children='Normalized Discounted Cumulative Gain (NDCG)'),
    html.Div(
        className="resource-link",
        children=[html.A("Link to scikit-learn", target="_blank", href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html")]
    ),
    html.H2(className='section-title', children='Overview'),
    html.Div(className='paragraph', children=[
        html.P("Once upon a time, there lived a man named Dave who was passionate about collecting rare coins. Dave spent his weekends attending coin auctions, searching for hidden treasures that would make his collection shine."),
        html.P("One day while Dave was reading a coin collecting magazine, he stumbled upon a new metric that tries to enhance the way items are ranked called \"Discounted Cumulative Gain\" (DCG). Curious and determined to understand this metric, he decided to relate it to his beloved hobby."),
        html.P("Dave envisioned his coin collection as a list of treasures, each with a distinct value. To better understand DCG, he decided to delve into its inner workings. First, he selected his top 10 rare coins and assigned each a relevance score based on historical significance, condition, and personal preference."),
        html.P("As Dave delved into DCG, he learned that it was a measure designed to evaluate the quality of a ranked list. It factored in both the relevance of each item and its position in the list. The formula for DCG involved summing up the relevance scores, but with a twist – the scores were discounted based on their position."),
        html.P("Dave imagined it like this: the further down the list a coin was, the less impact its relevance would have on the overall DCG. It was as if the treasures at the top of the list were more influential in determining the overall quality."),
        html.P("However, as Dave continued his exploration, he encountered a dilemma. What if he compared two lists of different lengths? The regular DCG wouldn't be fair in this scenario, as longer lists inherently had higher scores."),
        html.P("This is where Normalized Discounted Cumulative Gain (NDCG) stepped in as the hero of fairness. Dave learned that NDCG is calculated by dividing the regular DCG by the ideal DCG, which represented the perfect ranking."),
        html.P("Dave realized that calculating the ideal DCG involved ranking his treasures based on their relevance scores and then summing up their discounted values. This became the benchmark against which the regular DCG was normalized, ensuring a fair comparison between lists of different lengths."),
        html.P("With this newfound understanding, Dave went home excited to reorganize his coin collection using NDCG. He carefully considered the rarity of each coin, its quality, and its age. As he made these changes, he realized that NDCG was not only applicable to his hobby but also to various scenarios where ranking and order mattered."),
    ]),
    html.H2(className='section-title', children='Summary'),
    html.Div(className='paragraph', children=[
        html.P("Normalized Discounted Cumulative Gain (NDCG) is a metric that evaluates the effectiveness of a ranking algorithm by considering both the relevance of items and their positions in the ranked list, providing a normalized score that ranges from 0 to 1, with higher values indicating better performance.")
    ]),
    html.H2(className='section-title', children='Visualizing NDCG'),
    html.Button('Generate New Data', id='button-new-data', n_clicks=0),
    html.Div(className='plot-parameters', children=[
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='# of ratings (2-20)'),
            dcc.Input(className='parameter-value', id='input-n-ratings', value=10, min=2, max=20, step=1, type='number'),
        ]),
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='k'),
            dcc.Input(className='parameter-value', id='input-k', value=10, min=2, max=20, step=1, type='number'),
        ]),
    ]),
    html.P(className='metric-value-text', children=[
        "NDCG = ",
        html.Div(id="ndcg", className='metric-value'),
    ]),
    html.Div(className='hint', children=[
        "Update the ranking values by clicking on a value, changing it, then hitting enter."
    ]),
    html.Div(className='tables-two', children=[
        html.Div(id='table-ranking', className='table-dash'),
        html.Div(id='table-ranking-ideal', className='table-dash'),
    ]),
    html.H2(className='section-title', children='Formula'),
    html.Div(className='paragraph', children=[
        dcc.Markdown(
            r"""

            $$NDCG@k = \frac{DCG@k}{IDCG@k}$$

            $$DCG@k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$

            $$IDCG@k = \sum_{i=1}^{k} \frac{2^{rel_{\sigma(i)}} - 1}{\log_2(i + 1)}$$

            Where:
            - $$NDCG@k$$: Normalized Discounted Cumulative Gain while considering the highest $$k$$ scores.
            - $$DCG@k$$: Discounted Cumulative Gain while considering the highest $$k$$ scores.
            - $$IDCG@k$$: Ideal Discounted Cumulative Gain while considering the highest $$k$$ scores.
            - $$\sum_{i=1}^{k}$$: Summation from $$i = 1$$ to k.
            - $$2^{rel_i} - 1$$: The relevancy score of the result at position $$i$$.  It is raised to the power of 2 to emphasize the importance of highly relevant items. The subtraction of 1 is a scaling factor that ensures the value is 0 when the relevance score is 0.
            - $$\log_2(i + 1)$$: The logarithm base 2 of the position $$i + 1$$.
            - $$2^{rel_{\sigma(i)}} - 1$$: The relevance of the i-th result in the ideal ranking (sorted by relevance).
            """,
            mathjax=True
        )
    ]),
    html.H2(className='section-title', children='Assumptions'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("Independence of Judgments: "), "The relevancy score of one item does not influence the relevance judgment of another."]),
        html.P(children=[html.Strong("Positional Importance: "), "The order of relevance scores is valued more than their absolute values, which is emphasized in the diminishing importance of items as they appear lower in the ranked list."]),
        html.P(children=[html.Strong("Ideal Ranking is Meaningful: "), "There is a meaningful ideal ranking that represents the best possible order of items based on their relevance."]),
    ]),
    html.H2(className='section-title', children='Examples'),
    html.Div(className='paragraph', children=[
        html.P("1. Evaluating the effectiveness of a search engine's algorithm by comparing the relevance of search results to user queries"),
        html.P("2. Assessing the quality of product recommendations by comparing the ranked list of suggested items to the user's preferences."),
    ]),
])


@callback(
    Output('table-ranking', 'children'),
    Input('button-new-data', 'n_clicks'),
    Input('input-n-ratings', 'value'),
)
def ranking_table(random_state: int, n_ratings: int) -> List[dash_table.DataTable]:
    """
    Generate a ranking table with random ratings.
    
    Parameters:
        random_state: The random seed for reproducibility.
        n_ratings: The number of ratings to generate.
    
    Returns:
        The generated ranking table.
    """
    np.random.seed(random_state)
    
    # Generate a list of random ratings
    ranking = np.random.randint(1, 10, size=n_ratings).tolist()
    
    # Create a Dash DataTable with the ranking data
    ranking_table = dash_table.DataTable(
        id='table-ranking-data',
        data=[{'ranking': relevancy_score} for relevancy_score in ranking],
        columns=[{'name': 'ranking', 'id': 'ranking'}],
        editable=True
    )

    return [ranking_table]


@callback(
    Output('table-ranking-ideal', 'children'),
    Output('ndcg', 'children'),
    Input('table-ranking-data', 'data'),
    Input('input-k', 'value'),
)
def ideal_ranking_table_and_ndcg(
    ranking: List[Dict[str, str]],
    k: int
) -> Tuple[List[dash_table.DataTable], float]:
    """
    Generates the ideal ranking table and NDCG score based on
    the given ranking data and the k highest scores to consider.

    Parameters:
        ranking: The ranking data.
        k: Number of highest scores to consider in NDCG.

    Returns:
        The ideal ranking table and the rounded NDCG score.
    """

    # Sort the rankings to get the ideal rankings
    ranking_ideal = sorted(ranking, key=lambda x: float(x['ranking']), reverse=True)

    # Generate the ideal ranking table
    ranking_ideal_table = dash_table.DataTable(
        data=ranking_ideal,
        columns=[
            {'name': 'ideal ranking', 'id': 'ranking'},
        ],
        # Apply styling when a row is selected
        style_data_conditional=[                
            {
                "if": {"state": "selected"},
                "backgroundColor": "white",
                "border": "1px solid rgb(211, 211, 211)",
            },
        ]
    )

    # Extract the ranking values to compute NDCG
    ranking_values = [float(x['ranking']) for x in ranking]
    ranking_ideal_values = [float(x['ranking']) for x in ranking_ideal]
    ndcg = ndcg_score([ranking_ideal_values], [ranking_values], k=k)
    
    return [ranking_ideal_table], round(ndcg, 3)
