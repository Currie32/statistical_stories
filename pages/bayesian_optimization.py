import numpy as np
import plotly.graph_objects as go
from dash import callback, dcc, html, Input, Output, register_page
from plotly.graph_objs._figure import Figure as plotly_figure

from bayes_opt import BayesianOptimization, UtilityFunction


register_page(__name__, path="/bayesian-optimization")

layout = html.Div(className='content', children=[
    html.H1(className='content-title', children='Bayesian Optimization'),
    html.Div(
        className="resource-link",
        children=[html.A("Link to a bayesian-optimization package on GitHub", target="_blank", href="https://github.com/bayesian-optimization/BayesianOptimization")]
    ),
    html.H2(className='section-title', children='Overview'),
    html.Div(className='paragraph', children=[
        html.P("Once upon a time, there was a curious guy named Dave who loved growing plants in his backyard. One day, he decided to try a new approach to improve his gardening skills. Instead of randomly testing different fertilizers and watering schedules, he heard about something called Bayesian optimization."),
        html.P("Intrigued, Dave imagined it as a clever gardening assistant that could help him find a great combination of factors to make his plants thrive. He started by picturing the various elements affecting his garden—like sunlight, water, and fertilizer—as a formula for plant success."),
        html.P("Dave's first step was to plant a few seeds and randomly assign different amounts of sunlight, water, and fertilizer to each group. After a while, he observed how each group of plants responded. Some grew tall and strong, while others wilted or turned yellow."),
        html.P("Now, here's where the power of Bayesian optimization kicked in. Instead of blindly trying more random combinations, Dave applied Bayesian optimization to use the information he gathered to make educated guesses about what might work best for his plants. The more he learned, the smarter his guesses became."),
        html.P("It was like having a smart friend who remembered all the previous gardening experiments and suggested tweaks based on that knowledge. Dave adjusted the sunlight, water, and fertilizer for the next round of plants, making improvements with each iteration."),
        html.P("As he continued this process, Dave noticed something incredible. The Bayesian optimization system was learning from his garden experiments and honing in on the optimal conditions for plant growth. It wasn't just trial and error anymore; it was a strategic and efficient way to discover the perfect recipe for a thriving garden."),
        html.P("By the end of the gardening season, Dave's backyard was a lush paradise, filled with vibrant, healthy plants. He marveled at how Bayesian optimization had transformed his hobby into a science, helping him unlock the secrets of a flourishing garden."),
    ]),
    html.H2(className='section-title', children='Summary'),
    html.Div(className='paragraph', children=[
        html.P("Bayesian optimization is an efficient optimization strategy that uses a probabilistic model to balance exploration and exploitation, iteratively selecting new points to evaluate in the search space to find the optimal parameters for an expensive, unknown function.")
    ]),
    html.H2(className='section-title', children='Visualizing the Bayesian Optimization Process'),
    html.Button('Generate New Data', id='button-new-data', n_clicks=0),
    html.Div(className='plot-parameters', children=[
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='init_points'),
            dcc.Input(className='parameter-value', id='input-init-points', value=1, min=0, max=1000, step=1, type='number'),
            html.Label(className='parameter-label', children='kappa'),
            dcc.Input(className='parameter-value', id='input-kappa', value=2.6, min=0.1, max=100, step=0.1, type='number'),
            html.Label(className='parameter-label', children='xi'),
            dcc.Input(className='parameter-value', id='input-xi', value=0.01, min=0, max=1, step=0.01, type='number'),
        ]),
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='n_iter'),
            dcc.Input(className='parameter-value', id='input-n-iter', value=1, min=0, max=1000, step=1, type='number'),
            html.Label(className='parameter-label', children='kappa_decay'),
            dcc.Input(className='parameter-value', id='input-kappa-decay', value=1, min=0, max=1000, step=0.1, type='number'),
        ]),
        html.Div(className='parameter', children=[
            html.Label(className='parameter-label', children='kind'),
            dcc.Dropdown(
                id='input-kind', value='ucb', clearable=False,
                options=[{'label': 'ucb', 'value': 'ucb'}, {'label': 'ei', 'value': 'ei'}, {'label': 'poi', 'value': 'poi'}]
            ),
            html.Label(className='parameter-label', children='kappa_decay_delay'),
            dcc.Input(className='parameter-value', id='input-kappa-decay-delay', value=0, min=0, max=1000, step=0.1, type='number'),
        ]),
    ]),
    html.Div(className='plot-full-width', children=[dcc.Graph(id='plot-bayiesan-optimization')]),
    html.H2(className='section-title', children='Parameters'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("init_points: "), "The number of initial exploration iterations."]),
        html.P(children=[html.Strong("n_iter: "), "The number of iterations to find the maximum value."]),
        html.Div([
            html.P(children=[html.Strong("kind: "), "The type of acquisition function that will be used to explore the parameter space. The options are:"]),
            html.Ul(children=[
                html.Li(["Upper confidence bound (", html.Strong("ucb"), "): Balances exploration and exploitation by considering both the mean and the uncertainty (variance) of the model. It aims to choose points with a high expected value and high uncertainty."]),
                html.Li(["Expected Improvement (", html.Strong("ei"), "): Calculates the expected improvement in the objective function over the current best observed value. It is a more continuous measure and tends to be smoother than PI."]),
                html.Li(["Probability of improvement (", html.Strong("poi"), "): Calculates the probability that the objective function improvement at a given point exceeds a certain threshold. It encourages exploration by favoring points with a high probability of improvement."])
            ])
        ]),
        html.P(children=[html.Strong("kappa: "), "Determines the degree to which the algorithm explores versus exploits. A higher value increases the emphasis on exploration, encouraging the algorithm to sample points with higher uncertainty."]),
        html.P(children=[html.Strong("kappa_decay: "), "A decay factor applied to kappa over time. It allows for a dynamic adjustment of the exploration-exploitation balance during the optimization process. As the optimization progresses, kappa_decay is multiplied by kappa, reducing its value and potentially shifting the algorithm's focus more towards exploitation."]),
        html.P(children=[html.Strong("kappa_decay_delay: "), "The number of iterations before the decay of kappa starts. Delaying the multiplication of the decay parameter allows the algorithm to perform some initial exploration before transitioning to a more exploitative strategy."]),
        html.P(children=[html.Strong("xi: "), "Used in the Expected Improvement (EI) acquisition function to control the balance between exploration and exploitation by adjusting the threshold for improvement that a potential evaluation point must surpass to be considered promising."]),
    ]),
    html.H2(className='section-title', children='Assumptions'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("1. Gaussian Process Prior: "), "The distribution of functions has a joint Gaussian distribution, which is represented by a Gaussian process. This assumption allows for the use of Gaussian processes to model the objective function and its uncertainty."]),
        html.P(children=[html.Strong("2. Black-Box Function: "), "The objective function being optimized is a black-box function, meaning that it may be expensive to evaluate, and its functional form is unknown."]),
        html.P(children=[html.Strong("3. Sequential Design Strategy: "), "A sequential design strategy for global optimization of black-box functions. It aims to minimize the number of objective function evaluations by intelligently selecting the next point to evaluate based on the information gathered from previous evaluations."]),
    ]),
    html.H2(className='section-title', children='Appropriate use cases'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("Hyperparameter Tuning: "), "Commonly used in hyperparameter tuning for machine learning models, especially when the objective function is expensive to evaluate."]),
        html.P(children=[html.Strong("Experimental Design: "), "When conducting experiments with physical systems or scientific simulations, Bayesian optimization can be used to find optimal experimental conditions or input parameters, minimizing the number of experiments needed."]),
        html.P(children=[html.Strong("Adaptive Monte Carlo: "), "Optimize the sampling process to efficiently estimate integrals or perform simulations."]),
        html.P(children=[html.Strong("Environmental Monitoring: "), "Optimize the placement of sensors or monitoring devices."]),
    ]),
    html.H2(className='section-title', children='Inappropriate use cases'),
    html.Div(className='paragraph', children=[
        html.P(children=[html.Strong("Simple and Inexpensive Objective Functions: "), "Bayesian optimization might be overkill in these cases. Other optimization methods that do not involve the construction of a surrogate model, such as grid search or random search, may be more straightforward and computationally efficient."]),
        html.P(children=[html.Strong("High-Dimensional Search Spaces: "), "Bayesian optimization becomes less effective as the dimensionality of the search space increases. In high-dimensional spaces, the surrogate model may struggle to accurately model the objective function, leading to suboptimal performance. Other optimization techniques, such as genetic algorithms, particle swarm optimization, or simulated annealing, might be more suitable."]),
        html.P(children=[html.Strong("Discrete or Combinatorial Optimization: "), "Bayesian optimization is originally designed for continuous optimization problems. While there are extensions for handling discrete variables, it might not be the most natural fit for purely discrete or combinatorial optimization problems. Other optimization techniques, like genetic algorithms or simulated annealing, might be more appropriate in such cases."]),
        html.P(children=[html.Strong("Very Noisy Objective Functions: "), "While Bayesian optimization can handle noisy evaluations, it may struggle if the objective function is extremely noisy, to the point where the signal is overwhelmed by the noise. In such cases, other optimization methods, such as random search, genetic algorithms, particle swarm optimization, or simulated annealing, might be more appropriate."]),
        html.P(children=[html.Strong("Global Optimization in Non-Smooth Spaces: "), "If the objective function has non-smooth characteristics (e.g., discontinuities, sharp peaks), Bayesian optimization may struggle to provide accurate models. Evolutionary algorithms or other global optimization methods that do not assume smoothness might be more appropriate."]),
        html.P(children=[html.Strong("Stochastic and Dynamic Environments: "), "Bayesian optimization assumes a static objective function. If the optimization problem involves a highly dynamic or stochastic environment, where the objective function changes frequently, other adaptive optimization methods, such as reinforcement learning, genetic algorithms, particule swarm optimization, or simulated annealing, might be more suitable."]),
    ]),
])


@callback(
    Output('plot-bayiesan-optimization', 'figure'),
    Input('button-new-data', 'n_clicks'),
    Input('input-init-points', 'value'),
    Input('input-n-iter', 'value'),
    Input('input-kind', 'value'),
    Input('input-kappa', 'value'),
    Input('input-kappa-decay', 'value'),
    Input('input-kappa-decay-delay', 'value'),
    Input('input-xi', 'value'),
)
def plot_bayesian_optimization(
    random_state: int,
    init_points: int,
    n_iter: int,
    kind: str,
    kappa: float,
    kappa_decay: float,
    kappa_decay_delay: float,
    xi: float,
) -> plotly_figure:
    """
    Generate a plot of the Bayesian optimization process.
    
    Args:
        random_state: The random seed for reproducibility.
        init_points: The number of initial points to sample randomly.
        n_iter: The number of iterations to optimize.
        kind: The acquisition function to use.
        kappa: The kappa parameter for the acquisition function.
        kappa_decay: The decay rate for the kappa parameter.
        kappa_decay_delay: The delay before starting kappa decay.
        xi: The xi parameter for the acquisition function.
    
    Returns:
        The plotly figure object.
    """

    def target_function(x: float) -> float:
        """
        Calculate the value of the target function for a given input.

        Parameters:
            x: The input value for which the target function is calculated.

        Returns:
            The value of the target function for the given input.
        """
        np.random.seed(random_state)
        var1 = np.random.uniform(1, 5)
        var2 = np.random.uniform(1, 10)
        var3 = np.random.uniform(1, 5)
        return np.exp(-(x - var1)**2) + np.exp(-(x - var2)**2/10) + 1 / (x**2 + var3)

    # Define the parameter space for optimization
    pbounds = {'x': (-2, 10)}
    x = np.linspace(-2, 10, 1000).reshape(-1, 1)
    y = target_function(x)

    # Create BayesianOptimization object
    optimizer = BayesianOptimization(f=target_function, pbounds=pbounds, random_state=random_state)
    acq_function = UtilityFunction(kind=kind, kappa=kappa, kappa_decay=kappa_decay, kappa_decay_delay=kappa_decay_delay, xi=xi)
    optimizer.maximize(init_points=init_points, n_iter=n_iter, acquisition_function=acq_function)

    # Extract the observed data
    x_obs = np.array([[res["params"]["x"]] for res in optimizer.res])
    y_obs = np.array([res["target"] for res in optimizer.res])

    # Fit a Gaussian Process to the observed data
    optimizer._gp.fit(x_obs, y_obs)
    mu, sigma = optimizer._gp.predict(x, return_std=True)
    
    # Flatten the arrays
    x = x.flatten()
    y = y.flatten()
    uncertainties = sigma.flatten()

    # Initialize a plot
    target_trace = go.Scatter(x=x, y=y, mode='lines', name='Target Function')
    path_trace = go.Scatter(x=x_obs.flatten(), y=y_obs, mode='markers', name='Optimization Path', opacity=0.6, marker=dict(color='black'))
    lower_bounds = go.Scatter(x=x, y=mu - 1.96 * uncertainties, mode='lines', line=dict(color='rgba(255,0,0,0.3)'), name='Lower Bounds')
    upper_bounds = go.Scatter(x=x, y=mu + 1.96 * uncertainties, mode='lines', line=dict(color='rgba(255,0,0,0.3)'), name='Upper Bounds', fill='tonexty', fillcolor='rgba(255,0,0,0.1)')

    layout = go.Layout(title='Bayesian Optimization', xaxis=dict(title='x', range=(-3, 11)), yaxis=dict(title='f(x)', range=(min(y) - 0.5, max(y) + 0.5)))
    fig = go.Figure(data=[target_trace, path_trace, lower_bounds, upper_bounds], layout=layout)

    return fig
