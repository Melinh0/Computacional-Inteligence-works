class SimpleReactiveAgent:
    """
    Agente que decide apenas com base na percepção atual.
    Regras condição-ação:
        se sujeira -> aspirar
        senão se frente livre -> avançar
        senão -> virar à direita
    """

    def __init__(self):
        self.name = "SimpleReactive"

    def choose_action(self, percept):
        dirt, blocked = percept
        if dirt:
            return 'suck'
        if not blocked:
            return 'move_forward'
        return 'turn_right'

    def reset(self):
        pass