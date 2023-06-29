from dash import html


layout_home = html.Div(className='content', children=[
    html.H1(
        className='content-title',
        children='Welcome to Statistical Stories!'
    ),
    html.H2(
        className='content-subtitle',
        children="Learn Statistics through Stories and Visualizations"
    ),
    html.Div(className='paragraph-home', children=[
        html.P(
            "Statistics, data science, and machine learning are useful tools to know, but can be difficult to learn. "
            "That's why I created Statistical Stories."
        ),
        html.P(
            "Here you'll find easy to understand stories that explain the technical concepts in simple terms. "
        ),
        html.P(
            "You'll also be able to use interactive plots that visualize how distributions, algorithms, statistical tests, "
            "and much more change as you alter their parameters."
        ),
        html.P(
            "I hope that you'll find this website to be a useful resources, but it's still under active development, "
            "so if you see anything wrong or want me to add something soon, send me an email at david.currie32@gmail.com."
        )
    ]),
])
