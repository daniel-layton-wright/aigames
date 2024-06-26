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
