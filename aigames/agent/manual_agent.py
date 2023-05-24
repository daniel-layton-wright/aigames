from .agent import Agent


class ManualAgent(Agent):
    def get_action(self, state, legal_actions) -> int:
        print('Legal actions:')
        print('\n'.join(f'{i}: {action}' for i, action in enumerate(legal_actions)))
        print('\n')
        i = input('Enter an action: ')

        return int(i)