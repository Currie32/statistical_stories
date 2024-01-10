from typing import Dict, List, Tuple

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, no_update, page_container
from dash.dependencies import Input, Output, State
from flask import send_from_directory

from footer import footer


app = Dash(
    __name__,
    use_pages=True,
    pages_folder="pages",
    external_stylesheets=[dbc.icons.BOOTSTRAP, dbc.themes.BOOTSTRAP]
)
app.config.suppress_callback_exceptions = True
server = app.server

@server.route('/robots.txt')
def serve_robots():
    return send_from_directory('.', 'robots.txt', mimetype='text/plain')


@server.route('/sitemap.xml')
def serve_sitemap():
    return send_from_directory('.', 'sitemap.xml', mimetype='application/xml')


app.index_string = """<!DOCTYPE html>
<html lang="en">
    <head>
        <!-- Global site tag (gtag.js) - Google Analytics -->
        <script async src="https://www.googletagmanager.com/gtag/js?id=G-977RPT7S59"></script>
        <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){dataLayer.push(arguments);}
            gtag('js', new Date());

            gtag('config', 'G-977RPT7S59');
        </script>
        <meta charset="UTF-8">
        <meta name="description" content="Learn about statistics and data sciences with stories and interactive visualizations.">
        <meta name="keywords" content="statistics, data science, probability, intro to statistics, stats">
        <meta property="og:title" content="Statistical Stories">
        <meta property="og:description" content="Learn about statistics and data sciences with stories and interactive visualizations.">
        <meta property="og:image" content="https://statisticalstories.xyz/assets/favicon.ico">
        <meta property="og:url" content="https://statisticalstories.xyz">
        <meta name="twitter:card" content="https://statisticalstories.xyz/assets/favicon.ico">
        <meta name="twitter:title" content="Statistical Stories">
        <meta name="twitter:description" content="Learn about statistics and data sciences with stories and interactive visualizations.">
        <meta name="twitter:image" content="https://statisticalstories.xyz/assets/favicon.ico">
        <meta name="google-adsense-account" content="ca-pub-4657073290295216">
        <link rel="canonical" href="https://statisticalstories.xyz">
        <meta name="robots" content="index, follow">
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""


app.layout = html.Div([
    html.Div(className='container', children=[
        html.Div(id='navbar', children=[
            html.Div(className='button-menu-and-title', children=[
                # className "bi bi-list" is required for the menu icon
                # className "menu-hidden" is replaced by "menu-visible" depending on
                # the screen size and interactions with the website
                html.Button(html.I(className="bi bi-list"), id="button-navbar", className='menu-hidden'),
                html.H1(
                    className='navbar-title',
                    children=dcc.Link("Statistical Stories", href="/", id="link-home"),
                ),
            ]),
            # "display: none" is also replaced depending on screen size and interactions
            html.Div(id='navbar-menus', className='menu-hidden', style={'display': 'none'}, children=[
                html.Div([
                    dcc.Input(id='search-bar', type='text', placeholder='Search...'),
                    html.Div(id='search-results', style={'padding': '10px'})
                ]),
                dbc.DropdownMenu(
                    children=[
                        dbc.DropdownMenuItem(dcc.Link("Bernoulli", href="/bernoulli")),
                        dbc.DropdownMenuItem(dcc.Link("Beta", href="/beta")),
                        dbc.DropdownMenuItem(dcc.Link("Negative Binomial", href="/negative-binomial")),
                        dbc.DropdownMenuItem(dcc.Link("Normal", href="/normal")),
                        dbc.DropdownMenuItem(dcc.Link("Poisson", href="/poisson")),
                    ],
                    nav=True,
                    in_navbar=True,
                    label="Distributions",
                ),
                dbc.DropdownMenu(
                    children=[
                        dbc.DropdownMenuItem(dcc.Link("Cosine Similarity", href="/cosine-similarity")),
                        dbc.DropdownMenuItem(dcc.Link("NDCG", href="/ndcg")),
                    ],
                    nav=True,
                    in_navbar=True,
                    label="Metrics",
                ),
                dbc.DropdownMenu(
                    children=[
                        dbc.DropdownMenuItem(dcc.Link("Bayesian Optimization", href="/bayesian-optimization")),
                    ],
                    nav=True,
                    in_navbar=True,
                    label="Optimization Strategies",
                ),
                dbc.DropdownMenu(
                    children=[
                        dbc.DropdownMenuItem(dcc.Link("K-Nearest Neighbors", href="/k-nearest-neighbors")),
                    ],
                    nav=True,
                    in_navbar=True,
                    label="Statistical Models",
                ),
                dbc.DropdownMenu(
                    children=[
                        dbc.DropdownMenuItem(dcc.Link("ANOVA", href="/anova")),
                        dbc.DropdownMenuItem(dcc.Link("Chi-square", href="/chi-square")),
                    ],
                    nav=True,
                    in_navbar=True,
                    label="Statistical Tests",
                ),
            ]),
        ]),
        html.Div(id='content-body-wrapper', children=[
            dcc.Store(id="screen-width-store"),
            dcc.Location(id='url', refresh=False),
            page_container,
            footer
        ]),
    ])
])


# Get the width of the screen when the website loads
app.clientside_callback(
    """
    function() {return window.innerWidth}
    """,
    Output("screen-width-store", "data"),
    Input('button-navbar', 'n_clicks'),
)


@app.callback(
    Output('navbar-menus', 'className'),
    Output('navbar-menus', 'style'),
    Output('button-navbar', 'className'),
    Output('content-body-wrapper', 'className'), 
    Input('url', 'pathname'),
    Input('button-navbar', 'n_clicks'),
    Input('screen-width-store', 'data'),
    State('navbar-menus', 'className'),
    State('button-navbar', 'className'),
    State('content-body-wrapper', 'className'),
)
def display_navbar_menu_toggle(
    pathname: str,  # Not used but required input to close menu
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


def read_page_content(page_file: str) -> str:
    """
    Load and read the contents of a web page.
    """
    with open(page_file, 'r') as f:
        content = f.read()
    return content

@app.callback(
    Output('search-results', 'children'),
    Output('search-results', 'style'),
    Output('url', 'pathname'),
    [Input('search-bar', 'value')],
    [State('url', 'pathname')],
)
def search_pages(
    search_query: str,
    current_page: str
) -> Tuple[List[dcc.Link], Dict[str, str], str]:
    """
    Return a list of pages that contain the search query.
    Update the current_page if a link to a different page is clicked on.
    Update the padding of the search-results div based on the number of results.
    """

    # Don't display the search-results div if there is no search query
    if not search_query:
        return None, {'padding': '0px'}, current_page

    # Pages of the website that can be searched
    pages = [
        {'title': 'Bernoulli distribution', 'path': 'pages/distribution_bernoulli.py', 'url': '/bernoulli'},
        {'title': 'Beta distribution', 'path': 'pages/distribution_beta.py', 'url': '/beta'},
        {'title': 'Negative binomial distribution', 'path': 'pages/distribution_negative_binomial.py', 'url': '/negative-binomial'},
        {'title': 'Normal distribution', 'path': 'pages/distribution_normal.py', 'url': '/normal'},
        {'title': 'Poisson distribution', 'path': 'pages/distribution_poisson.py', 'url': '/poisson'},
        {'title': 'Cosine similarity', 'path': 'pages/metric_cosine_similarity.py', 'url': '/cosine-similarity'},
        {'title': 'NDCG', 'path': 'pages/metric_ndcg.py', 'url': '/ndcg'},
        {'title': 'Bayesian optimization', 'path': 'pages/bayesian_optimization.py', 'url': '/bayesian-optimization'},
        {'title': 'k-nearest neighbors', 'path': 'pages/model_k_nearest_neighbors.py', 'url': '/k-nearest-neighbors'},
        {'title': 'ANOVA', 'path': 'pages/test_anova.py', 'url': '/anova'},
        {'title': 'Chi-square', 'path': 'pages/test_chi_square.py', 'url': '/chi-square'},
    ]

    # Append pages that match the search query
    matching_pages = []
    for page in pages:
        content = read_page_content(page['path'])
        if search_query.lower() in content.lower():
            matching_pages.append(page)

    # If there are matching pages, display them
    if matching_pages:
        result_list = []
        for page in matching_pages:
            result_list.append(dcc.Link(page['title'], href=page['url']))
        return result_list, {'padding': '10px'}, no_update
    
    # If no matching pages are found, display a message
    else:
        return html.P('No matching pages found.'), {'padding': '10px 10px 1px'}, no_update


@app.callback(
    Output('search-bar', 'value'),
    Input('search-results', 'n_clicks'),
    State('url', 'pathname'),
)
def reset_search_bar_text(clicks, url):
    """
    Reset the search bar's text when a result is clicked on.
    """
    return ""
        

if __name__ == '__main__':
    app.run_server(debug=True, port=8080)
