import numpy as np
from itertools import product

def softmax(x1, x2):
    """
    Computes a soft maximum of both arguments.
    In case `x1` and `x2` are arrays, computes the element-wise softmax.
    Args:
        x1: Scalar or ndarray.
        x2: Scalar or ndarray.
    Returns:
        The soft maximum of the given arguments, either scalar or ndarray,
        depending on the input.
    """
    x_max = np.maximum(x1, x2)
    x_min = np.minimum(x1, x2)
    return x_max + np.log(1.0 + np.exp(x_min - x_max))

def local_causal_action_probabilities(p_transition, terminal, reward, discount, eps=1e-5):
    """
    Compute the local action probabilities (policy) required for the edge
    frequency calculation for maximum causal entropy reinfocement learning.
    This is Algorithm 9.1 from Ziebart's thesis (2010) combined with
    discounting for convergence reasons as proposed in the same thesis.
    Args:
        p_transition: The transition probabilities of the MDP as table
            `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            specifying the probability of a transition from state `from` to
            state `to` via action `action` to succeed.
        terminal: Either the terminal reward function or a collection of
            terminal states. Iff `len(terminal)` is equal to the number of
            states, it is assumed to contain the terminal reward function
            (phi) as specified in Ziebart's thesis. Otherwise `terminal` is
            assumed to be a collection of terminal states from which the
            terminal reward function will be derived.
        reward: The reward signal per state as table
            `[state: Integer] -> reward: Float`.
        discount: A discounting factor as Float.
        eps: The threshold to be used as convergence criterion for the state
            partition function. Convergence is assumed if the state
            partition function changes less than the threshold on all states
            in a single iteration.
    Returns:
        The local action probabilities (policy) as map
        `[state: Integer, action: Integer] -> probability: Float`
    """
    n_states, _, n_actions = p_transition.shape

    # set up terminal reward function
    if len(terminal) == n_states:
        reward_terminal = np.array(terminal, dtype=np.float)
    else:
        reward_terminal = -np.inf * np.ones(n_states)
        reward_terminal[terminal] = 0.0

    # set up transition probability matrices
    p = [np.array(p_transition[:, :, a]) for a in range(n_actions)]

    # compute state log partition V and state-action log partition Q
    v = -1e200 * np.ones(n_states)  # np.dot doesn't behave with -np.inf

    delta = np.inf
    while delta > eps:
        v_old = v

        q = np.array([reward + discount * p[a].dot(v_old) for a in range(n_actions)]).T

        v = reward_terminal
        for a in range(n_actions):
            v = softmax(v, q[:, a])

        # for some reason numpy chooses an array of objects after reduction, force floats here
        v = np.array(v, dtype=np.float)

        delta = np.max(np.abs(v - v_old))

    # compute and return policy
    return np.exp(q - v[:, None])

def compute_expected_causal_svf(p_transition, p_initial, terminal, reward, discount,
                                eps_lap=1e-5, eps_svf=1e-5):
    """
    Compute the expected state visitation frequency for maximum causal
    entropy IRL.
    This is a combination of Algorithm 9.1 and 9.3 of Ziebart's thesis
    (2010). See `local_causal_action_probabilities` and
    `expected_svf_from_policy` for more details.
    Args:
        p_transition: The transition probabilities of the MDP as table
            `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            specifying the probability of a transition from state `from` to
            state `to` via action `action` to succeed.
        p_initial: The probability of a state being an initial state as map
            `[state: Integer] -> probability: Float`.
        terminal: Either the terminal reward function or a collection of
            terminal states. Iff `len(terminal)` is equal to the number of
            states, it is assumed to contain the terminal reward function
            (phi) as specified in Ziebart's thesis. Otherwise `terminal` is
            assumed to be a collection of terminal states from which the
            terminal reward function will be derived.
        reward: The reward signal per state as table
            `[state: Integer] -> reward: Float`.
        discount: A discounting factor as Float.
        eps_lap: The threshold to be used as convergence criterion for the
            state partition function. Convergence is assumed if the state
            partition function changes less than the threshold on all states
            in a single iteration.
        eps_svf: The threshold to be used as convergence criterion for the
            expected state-visitation frequency. Convergence is assumed if
            the expected state visitation frequency changes less than the
            threshold on all states in a single iteration.
    """
    p_action = local_causal_action_probabilities(p_transition, terminal, reward, discount, eps_lap)
    return expected_svf_from_policy(p_transition, p_initial, terminal, p_action, eps_svf)


def irl_causal(p_transition, features, terminal, trajectories, optim, init, discount,
               eps=1e-4, eps_svf=1e-5, eps_lap=1e-5):
    """
    Compute the reward signal given the demonstration trajectories using the
    maximum causal entropy inverse reinforcement learning algorithm proposed
    Ziebart's thesis (2010).
    Args:
        p_transition: The transition probabilities of the MDP as table
            `[from: Integer, to: Integer, action: Integer] -> probability: Float`
            specifying the probability of a transition from state `from` to
            state `to` via action `action` to succeed.
        features: The feature-matrix (e.g. as numpy array), mapping states
            to features, i.e. a matrix of shape (n_states x n_features).
        terminal: Either the terminal reward function or a collection of
            terminal states. Iff `len(terminal)` is equal to the number of
            states, it is assumed to contain the terminal reward function
            (phi) as specified in Ziebart's thesis. Otherwise `terminal` is
            assumed to be a collection of terminal states from which the
            terminal reward function will be derived.
        trajectories: A list of `Trajectory` instances representing the
            expert demonstrations.
        optim: The `Optimizer` instance to use for gradient-based
            optimization.
        init: The `Initializer` to use for initialization of the reward
            function parameters.
        discount: A discounting factor for the log partition functions as
            Float.
        eps: The threshold to be used as convergence criterion for the
            reward parameters. Convergence is assumed if all changes in the
            scalar parameters are less than the threshold in a single
            iteration.
        eps_lap: The threshold to be used as convergence criterion for the
            state partition function. Convergence is assumed if the state
            partition function changes less than the threshold on all states
            in a single iteration.
        eps_svf: The threshold to be used as convergence criterion for the
            expected state-visitation frequency. Convergence is assumed if
            the expected state visitation frequency changes less than the
            threshold on all states in a single iteration.
    """
    n_states, _, n_actions = p_transition.shape
    _, n_features = features.shape

    # compute static properties from trajectories
    e_features = feature_expectation_from_trajectories(features, trajectories)
    p_initial = initial_probabilities_from_trajectories(n_states, trajectories)

    # basic gradient descent
    theta = init(n_features)
    delta = np.inf

    optim.reset(theta)
    while delta > eps:
        theta_old = theta.copy()

        # compute per-state reward
        reward = features.dot(theta)

        # compute the gradient
        e_svf = compute_expected_causal_svf(p_transition, p_initial, terminal, reward, discount,
                                            eps_lap, eps_svf)

        grad = e_features - features.T.dot(e_svf)

        # perform optimization step and compute delta for convergence
        optim.step(grad)
        delta = np.max(np.abs(theta_old - theta))

    # re-compute per-state reward and return
    return features.dot(theta)