from typing import Dict, Tuple

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, no_update
from dash.dependencies import Input, Output, State
from plotly.graph_objs._figure import Figure as plotly_figure

from pages.distributions.normal import layout_normal, distribution_plots_normal, histogram_normal_plot
from pages.distributions.poisson import layout_poisson, distribution_plots_poisson, histogram_poisson_plot
from pages.statistical_models.k_nearest_neighbors import layout_k_nearest_neighbors, k_nearest_neighbors
from pages.statistical_tests.anova import layout_anova, anova
from pages.home import layout_home


app = Dash(
    __name__,
    # Add stylesheet for menu icon with small screens
    external_stylesheets=[dbc.icons.BOOTSTRAP]
)
app.config.suppress_callback_exceptions = True
server = app.server


app.layout = html.Div([
    html.Div(className='container', children=[
        html.Div(id='navbar', children=[
            html.Div(className='button-menu-and-title', children=[
                # className "bi bi-list" is required for the menu icon
                # className "menu-hidden" is replaced by "menu-visible" depending on
                # the screen size and interactions
                html.Button(html.I(className="bi bi-list"), id="button-menu", className='menu-hidden'),
                html.H1(
                    className='navbar-title',
                    children=html.A('Statistical Stories', id="link-home", href="/")
                ),
            ]),
            # "display: none" is also replaced depending on screen size and interactions
            html.Div(id='navbar-menus', className='menu-hidden', style={'display': 'none'}, children=[
                html.Div([
                    dcc.Input(id='search-bar', type='text', placeholder='Search...'),
                    html.Div(id='search-results')
                ]),
                dcc.Dropdown(
                    className='navbar-menu',
                    id='menu-distributions',
                    options=[
                        {'label': 'Gamma Distribution', 'value': '/gamma'},
                        {'label': 'Normal Distribution', 'value': '/normal'},
                        {'label': 'Poisson Distribution', 'value': '/poisson'},
                        {'label': 'Student-T Distribution', 'value': '/student-t'},
                    ],
                    placeholder='Distributions',
                    clearable=False,
                ),
                dcc.Dropdown(
                    className='navbar-menu',
                    id='menu-statistical-models',
                    options=[
                        {'label': 'K-Nearest Neighbors', 'value': '/k-nearest-neighbors'},
                        {'label': 'Linear Regression', 'value': '/linear-regression'},
                        {'label': 'Support Vector Machine', 'value': '/support-vector-machine'},
                    ],
                    placeholder='Statistical Models',
                    clearable=False,
                ),
                dcc.Dropdown(
                    className='navbar-menu',
                    id='menu-statistical-tests',
                    options=[
                        {'label': 'ANOVA', 'value': '/anova'},
                        {'label': 'Chi-square test', 'value': '/chi-square'},
                        {'label': 't-test', 'value': '/t-test'},
                        {'label': 'Wilcoxon rank-sum test', 'value': '/wilcoxon-rank-sum'},
                    ],
                    placeholder='Statistical Tests',
                    clearable=False,
                ),
            ]),
        ]),
        html.Div(id='content-body-wrapper', children=[
            dcc.Store(id="screen-width-store"),
            dcc.Location(id='url', refresh=False),
            html.Div(id='content-body', className='menu-hidden')
        ])
    ])
])


@app.callback(
    Output('url', 'pathname', allow_duplicate=True),
    Output('menu-distributions', 'value', allow_duplicate=True),
    Output('menu-statistical-tests', 'value', allow_duplicate=True),
    Output('menu-statistical-models', 'value', allow_duplicate=True),
    Input('menu-distributions', 'value'),
    Input('menu-statistical-tests', 'value'),
    Input('menu-statistical-models', 'value'),
    prevent_initial_call=True,
)
def update_url(
    menu_value_distributions: str,
    menu_value_statistical_tests: str,
    menu_value_statistical_models: str,
) -> Tuple[str, str, str, str]:
    """
    Update the URL using the value selected from one of the menus.
    """
    if menu_value_distributions != "Distributions":
        return menu_value_distributions, 'Distributions', "Statistical Tests", "Statistical Models"

    elif menu_value_statistical_models != "Statistical Models":
        return menu_value_statistical_models, 'Distributions', "Statistical Tests", "Statistical Models"

    else:
        return menu_value_statistical_tests, 'Distributions', "Statistical Tests", "Statistical Models"


@app.callback(
    Output('content-body', 'children', allow_duplicate=True),
    [Input('url', 'pathname')],
    prevent_initial_call=True,
)
def scroll_to_top(pathname):
    """
    Scroll to the top of the page when a new URL is used.
    """
    scroll_script = '''
        <script>
            window.scrollTo(0, 0);
        </script>
    '''
    return scroll_script


@app.callback(
    Output('content-body', 'children'),
    Input('url', 'pathname'),
)
def update_content(pathname: str) -> str:
    """
    Update the content of the website to reflect the new url.
    """
    if pathname == "/":
        return layout_home
    # Distributions
    elif pathname == '/normal':
        return layout_normal
    elif pathname == '/poisson':
        return layout_poisson

    # Statistical Models
    elif pathname == '/k-nearest-neighbors':
        return layout_k_nearest_neighbors
    
    # Statistical Tests
    elif pathname == '/anova':
        return layout_anova

    # Page not found
    else:
        return html.Div([
            html.H1('404: Not found'),
            html.P(f'The requested URL "{pathname}" was not found on this server.')
        ])


# Get the width of the screen when the website loads
app.clientside_callback(
    """
    function() {return window.innerWidth}
    """,
    Output("screen-width-store", "data"),
    Input('button-menu', 'n_clicks'),
)


@app.callback(
    Output('navbar-menus', 'className', allow_duplicate=True),
    Output('navbar-menus', 'style'),
    Output('button-menu', 'className'),
    Output('content-body-wrapper', 'className'), 
    Input('menu-distributions', 'value'),
    Input('menu-statistical-models', 'value'),
    Input('menu-statistical-tests', 'value'),
    Input('button-menu', 'n_clicks'),
    Input('screen-width-store', 'data'),
    State('navbar-menus', 'className'),
    State('button-menu', 'className'),
    State('content-body-wrapper', 'className'),
    prevent_initial_call=True,
)
def display_navbar_menu_toggle(
    menu_value_distributions: str,  # Not used but required input to close menu
    menu_value_statistical_models: str,  # Not used but required input to close menu
    menu_value_statistical_tests: str,  # Not used but required input to close menu
    n_clicks: int,
    screen_width: int,
    menu_classname: str,
    button_class_name: str,
    content_class_name: str,
) -> Tuple[str, Dict[str, str], str, str]:
    """
    Logic of whether to display the dropdown navbar menu.
    """
    if screen_width < 800 and not n_clicks:
        return 'menu-hidden', {'display': 'none'}, 'menu-hidden', 'menu-hidden'
    elif menu_classname == 'menu-visible' and n_clicks:
        return 'menu-hidden', {'display': 'none'}, 'menu-hidden', 'menu-hidden'
    elif button_class_name == 'menu-hidden' and n_clicks:
        return 'menu-visible', {'display': 'block'}, 'menu-visible', 'menu-visible'
    elif content_class_name == 'menu-hidden':
        return 'menu-hidden', {'display': 'none'}, 'menu-hidden', 'menu-hidden'
    else:
        return 'menu-visible', {'display': 'block'}, 'menu-visible', 'menu-visible'


@app.callback(
    Output('normal-plot', 'figure'),
    Output('normal-cdf-plot', 'figure'),
    Input('normal-mean-input', 'value'),
    Input('normal-std-input', 'value'),
)
def update_distribution_plots_normal(
    mean: float,
    std: float,
) -> Tuple[plotly_figure, plotly_figure]:
    """
    Update the normal distributions plots using the
    new mean and standard deviation.
    """
    plot_pdf, plot_cdf = distribution_plots_normal(mean, std)
    return plot_pdf, plot_cdf


@app.callback(Output('poisson-plot', 'figure'),
              Output('poisson-cdf-plot', 'figure'),
              Input('poisson-lambda-input', 'value'))
def update_distribution_plots_poisson(lam: float) -> plotly_figure:
    """
    Update the poisson distribution plots using the new lambda value.
    """
    plot_pmf, plot_cmf = distribution_plots_poisson(lam)
    return plot_pmf, plot_cmf


@app.callback(
    Output('histogram-plot', 'figure'),
    Input('button-new-data', 'n_clicks'),
    Input('url', 'pathname'),
)
def generate_histogram_plot(n_clicks: int, pathname: str) -> Tuple[plotly_figure, str]:
    """
    Generate a histogram based on the pathname when the "Generate New Data" button is clicked.
    """
    if pathname == "/normal":
        return histogram_normal_plot()

    elif pathname == "/poisson":
        return histogram_poisson_plot()


@app.callback(
    Output('knn-plot', 'figure'),
    Input('knn-n-neighbors-input', 'value'),
    Input('knn-weights-input', 'value'),
    Input('knn-algorithm-input', 'value'),
    Input('button-new-data', 'n_clicks'),
)
def update_k_nearest_neighbors_plot(
    n_neighbors: int,
    weights: str,
    algorithm: str,
    random_state: int,
) -> plotly_figure:
    """
    Update the poisson distribution plots using the new lambda value.
    """
    plot = k_nearest_neighbors(n_neighbors, weights, algorithm, random_state)
    return plot


@app.callback(
    Output('anova-plot', 'figure'),
    Output('anova-table', 'children'),
    Input('button-new-data', 'n_clicks'),
    Input('mean-group-a', 'value'),
    Input('std-group-a', 'value'),
    Input('samples-group-a', 'value'),
    Input('mean-group-b', 'value'),
    Input('std-group-b', 'value'),
    Input('samples-group-b', 'value'),
    Input('mean-group-c', 'value'),
    Input('std-group-c', 'value'),
    Input('samples-group-c', 'value'),
)
def update_anova_plot(
    n_clicks,
    mean_a, std_dev_a, samples_a,
    mean_b, std_dev_b, samples_b,
    mean_c, std_dev_c, samples_c,
):
    """
    qwe
    """
    plot, table = anova(n_clicks, mean_a, std_dev_a, samples_a, mean_b, std_dev_b, samples_b, mean_c, std_dev_c, samples_c)

    return plot, table


# Function to read the content of a page from a file
def read_page_content(page_file):
    with open(page_file, 'r') as f:
        content = f.read()
    return content

@app.callback(
    Output('search-results', 'children'),
    Output('url', 'pathname'),
    [Input('search-bar', 'value')],
    [State('url', 'pathname')],
)
def search_pages(search_term, current_page):
    # Perform the search logic here based on the search term
    # You can search through your pages or content data structure to find relevant pages
    # For demonstration purposes, let's assume you have a list of page files

    if not search_term:
        return None, current_page

    pages = [
        {'title': 'Normal distribution', 'path': 'pages/distributions/normal.py', 'url': '/normal'},
        {'title': 'Poisson distribution', 'path': 'pages/distributions/poisson.py', 'url': '/poisson'},
        {'title': 'k-Nearest Neighbors', 'path': 'pages/statistical_models/k_nearest_neighbors.py', 'url': '/k-nearest-neighbors'},
        {'title': 'ANOVA', 'path': 'pages/statistical_tests/anova.py', 'url': 'anova'},
    ]
    matching_pages = []

    for page in pages:
        content = read_page_content(page['path'])
        if search_term.lower() in content.lower():
            matching_pages.append(page)

    if matching_pages:
        # If there are matching pages, display them
        result_list = []
        for page in matching_pages:
            result_list.append(html.A(page['title'], href=page['url']))
        return result_list, no_update
    else:
        # If no matching pages are found, display a message
        return html.P('No matching pages found.'), no_update
        


if __name__ == '__main__':
    app.run_server(debug=True)
