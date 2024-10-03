from aigames.mcts.mcts import MCTS


def graph_from_mcts(mcts: MCTS, root_index: int = 0):
    """
    Create a graph from an MCTS object

    :param mcts: The MCTS object
    :param root_index: The index of the root node to use
    :return: A graphviz graph
    """
    r = root_index
    import pygraphviz
    import re

    def node_to_str(node):
        state_str = mcts.states[r, node] if not hasattr(mcts.game, 'state_to_str') else mcts.game.state_to_str(
            mcts.states[r, node])
        state_str = re.sub(
            r'[\u001B\u009B][\[\]()#;?]*((([a-zA-Z\d]*(;[-a-zA-Z\d\/#&.:=?%@~_]*)*)?\u0007)'
            r'|((\d{1,4}(?:;\d{0,4})*)?[\dA-PR-TZcf-ntqry=><~]))',
            '', state_str)

        legal_actions = mcts.legal_actions_mask[r, node]

        return (f'{state_str}\n'
                f'v: {mcts.values[r, node]}\n'
                f'pi: {mcts.pi[r, node][legal_actions]}\n'
                f'N: {mcts.n[r, node][legal_actions]}\n'
                f'W: {mcts.w[r, node][legal_actions]}\n'
                f'Q: {(mcts.w[r, node] / mcts.n[r, node].unsqueeze(-1))[legal_actions]}\n')

    def action_to_str(action):
        return mcts.game.action_to_str(action) if hasattr(mcts.game, 'action_to_str') else str(action)

    graph = pygraphviz.AGraph(directed=True)

    graph.add_node(1, label=node_to_str(1), justify='left')

    q = mcts.next_idx[r, 1].tolist()
    while len(q) > 0:
        node = q.pop(0)

        if node == 0:
            continue

        graph.add_node(node, label=node_to_str(node), justify='left')
        graph.add_edge(mcts.parent_nodes[r, node].item(), node, label=action_to_str(mcts.actions[r, node]))
        q.extend(mcts.next_idx[r, node].tolist())

    return graph


def make_mcts_plots(mcts: MCTS, n_iters: int = None, ax=None):
    """
    Run MCTS search and create plots of visit counts, Q-values (scaled and raw), and pi distribution for root actions.

    :param mcts: The MCTS object
    :param n_iters: Number of iterations to run (if None, use mcts.n_iters)
    :param ax: Optional list of 4 matplotlib axes to plot on. If None, new axes will be created.
    :return: Matplotlib figure with plots
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n_iters = n_iters or mcts.n_iters
    n_actions = mcts.n_actions
    
    # Expand the root node if it's a leaf
    if mcts.is_leaf[0, 1]:
        mcts.expand()

    # Get legal actions mask for the root node
    legal_actions_mask = mcts.legal_actions_mask[0, 1].cpu().numpy()
    legal_actions = np.where(legal_actions_mask)[0]

    # Initialize arrays to store statistics for legal actions only
    visit_counts = np.zeros((n_iters, len(legal_actions)))
    q_values_scaled = np.zeros((n_iters, len(legal_actions)))
    q_values_raw = np.zeros((n_iters, len(legal_actions)))
    pi_values = np.zeros((n_iters, len(legal_actions)))

    # Run MCTS search and collect statistics
    for i in range(n_iters):
        mcts.search_for_n_iters(i)  # this is the total to reach
        visit_counts[i] = mcts.n[0, 1, legal_actions].cpu().numpy()
        q_values_scaled[i] = mcts.Q([0], [1], scale_Q=True)[0, legal_actions].cpu().numpy()
        q_values_raw[i] = mcts.Q([0], [1], scale_Q=False)[0, legal_actions].cpu().numpy()
        _, pi = mcts.action_selector.get_final_actions_and_pi(mcts, tau=1.0)
        pi_values[i] = pi[0, legal_actions].cpu().numpy()
    
    # Get action labels
    action_labels = ([mcts.game.action_to_str(action) for action in legal_actions] 
                     if hasattr(mcts.game, 'action_to_str') 
                     else [f'Action {action}' for action in legal_actions])

    # Create plots
    if ax is None:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 24))
    else:
        fig = ax[0].figure
        ax1, ax2, ax3, ax4 = ax

    # Plot visit counts
    for idx, action_label in enumerate(action_labels):
        ax1.plot(range(1, n_iters + 1), visit_counts[:, idx], label=action_label)
    ax1.set_xlabel('Search steps')
    ax1.set_ylabel('Visit count')
    ax1.set_title('Visit Counts for Legal Root Actions')
    ax1.legend()
    ax1.grid(True)

    # Plot scaled Q-values
    for idx, action_label in enumerate(action_labels):
        ax2.plot(range(1, n_iters + 1), q_values_scaled[:, idx], label=action_label)
    ax2.set_xlabel('Search steps')
    ax2.set_ylabel('Scaled Q-value')
    ax2.set_title('Scaled Q-values for Legal Root Actions')
    ax2.legend()
    ax2.grid(True)

    # Plot raw Q-values
    for idx, action_label in enumerate(action_labels):
        ax3.plot(range(1, n_iters + 1), q_values_raw[:, idx], label=action_label)
    ax3.set_xlabel('Search steps')
    ax3.set_ylabel('Raw Q-value')
    ax3.set_title('Raw Q-values for Legal Root Actions')
    ax3.legend()
    ax3.grid(True)

    # Plot pi distribution
    for idx, action_label in enumerate(action_labels):
        ax4.plot(range(1, n_iters + 1), pi_values[:, idx], label=action_label)
    ax4.set_xlabel('Search steps')
    ax4.set_ylabel('Probability')
    ax4.set_title('Final Pi Distribution for Legal Root Actions')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()

    return fig