from dash import dcc, html


layout_home = html.Div(className='content', children=[
    html.H1(
        className='home-title',
        children='Welcome to the Statistical Stories!'
    ),
    html.Div(children='''
        Explaining statistical concepts using stories.
    '''),
])
