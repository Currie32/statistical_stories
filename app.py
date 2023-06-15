import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from pages.distributions.normal import normal_layout, update_normal_plot, plot_normal_distribution_with_pdf
from pages.distributions.poisson import poisson_layout, update_poisson_plot
from pages.home import home_layout


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.icons.BOOTSTRAP]
)
app.config.suppress_callback_exceptions = True

app.layout = html.Div([

    html.Div(className='container', children=[
        html.Div(id='navbar', children=[
            html.Div(className='button-and-title', children=[
                html.Button(html.I(className="bi bi-list"), id="button-menu", className='menu-hidden'),
                html.H1(
                    className='navbar-title',
                    children=html.A('Statistical Stories', id="link-home", href="/")
                ),
            ]),
            html.Div(id='navbar-menus', className='menu-hidden', style={'display': 'none'}, children=[
                dcc.Dropdown(
                    className='navbar-menu-item',
                    id='menu-distributions',
                    options=[
                        {'label': 'Normal Distribution', 'value': '/normal'},
                        {'label': 'Student-T Distribution', 'value': '/student-t'},
                        {'label': 'Poisson Distribution', 'value': '/poisson'},
                        {'label': 'Gamma Distribution', 'value': '/gamma'},
                    ],
                    placeholder='Distributions',
                    clearable=False,
                ),
                dcc.Dropdown(
                    className='navbar-menu-item',
                    id='menu-statistical-tests',
                    options=[
                        {'label': 'ANOVA', 'value': '/anova'},
                        {'label': 'Chi-square test', 'value': '/chi-square'},
                        {'label': 't-test', 'value': '/t-test'},
                        {'label': 'Wilcoxon rank-sum test', 'value': '/wilcoxon-rank-sum'},
                    ],
                    placeholder='Statistical Tests',
                ),
                dcc.Dropdown(
                    className='navbar-menu-item',
                    id='menu-statistical-models',
                    options=[
                        {'label': 'Linear Regression', 'value': '/linear-regression'},
                        {'label': 'Support Vector Machine', 'value': '/support-vector-machine'},
                        {'label': 'K-Nearest Neighbours', 'value': '/k-nearest-neighbours'},
                    ],
                    placeholder='Statistical Models',
                ),
            ]),
        ]),
        html.Div(id='content-body', children=[
            dcc.Store(id="screen-width-store"),
            dcc.Location(id='url', refresh=False),
            html.Div(id='page-content', className='menu-hidden')
        ])
    ])
])

@app.callback(
    Output('url', 'pathname'),
    Input('menu-distributions', 'value'),
    Input('menu-statistical-tests', 'value'),
    Input('menu-statistical-models', 'value'),
    prevent_initial_call=True,
)
def update_url(value, a, b):
    return value

@app.callback(
    Output('page-content', 'children', allow_duplicate=True),
    Input('url', 'pathname'),
    prevent_initial_call=True
)
def display_page(pathname):
    if pathname == "/":
        return home_layout
    elif pathname == '/normal':
        return normal_layout
    elif pathname == '/poisson':
        return poisson_layout
    else:
        return html.Div([
            html.H1('404: Not found'),
            html.P(f'The requested URL "{pathname}" was not found on this server.')
        ])


@app.callback(Output('normal-plot', 'figure'),
              Output('normal-cdf-plot', 'figure'),
              Input('normal-mean-input', 'value'),
              Input('normal-std-input', 'value'))
def update_normal(mean, std):
    fig_pdf, fig_cdf = update_normal_plot(mean, std)
    return fig_pdf, fig_cdf


@app.callback(Output('poisson-plot', 'figure'),
              Output('poisson-cdf-plot', 'figure'),
              Input('poisson-lambda-input', 'value'))
def update_poisson(lam):
    fig_pmf, fig_cmf = update_poisson_plot(lam)
    return fig_pmf, fig_cmf

@app.callback(
    Output('histogram-plot', 'figure'),
    Output('text-test-result', 'children'),
    Input('button-new-data', 'n_clicks')
)
def update_plot_normal_distribution_with_pdf(url):
    fig, text = plot_normal_distribution_with_pdf()
    return fig, text



app.clientside_callback(
    """
    function() {
        return window.innerWidth;
    }
    """,
    Output("screen-width-store", "data"),
    Input('button-menu', 'n_clicks'),
)


@app.callback(
    Output('navbar-menus', 'className', allow_duplicate=True),
    Output('navbar-menus', 'style'),
    Output('button-menu', 'className'),
    Output('content-body', 'className'),
    Output('menu-distributions', 'value'),
    Input('menu-distributions', 'value'),
    Input('button-menu', 'n_clicks'),
    Input('screen-width-store', 'data'),
    State('navbar-menus', 'className'),
    State('button-menu', 'className'),
    State('content-body', 'className'),
    prevent_initial_call=True,
)
def toggle_menu(value1, n_clicks, screen_width, menu_classname, button_class_name, content_class_name):

    if screen_width >= 800:
        return 'menu-visible', {'display': 'block'}, 'menu-hidden', 'menu-hidden', value1
    elif menu_classname == 'menu-visible':
        return 'menu-hidden', {'display': 'none'}, 'menu-hidden', 'menu-hidden', value1
    elif button_class_name == 'menu-hidden' and n_clicks:
        return 'menu-visible', {'display': 'block'}, 'menu-visible', 'menu-visible', value1
    elif content_class_name == 'menu-hidden':
        return 'menu-hidden', {'display': 'none'}, 'menu-hidden', 'menu-hidden', value1
    else:
        return 'menu-visible', {'display': 'block'}, 'menu-visible', 'menu-visible', value1



if __name__ == '__main__':
    app.run_server(debug=True)

