import random

class Environment:
    """
    Ambiente de grade retangular com sujeira e obstáculos.
    Parcialmente observável: o agente só sabe se a célula atual está suja
    e se há um obstáculo imediatamente à frente.
    Determinístico: ações têm efeito único.
    """

    def __init__(self, width, height, dirt_prob=0.3, obstacle_prob=0.1):
        self.width = width
        self.height = height
        self.dirt_prob = dirt_prob
        self.obstacle_prob = obstacle_prob
        self.grid = None          # True = sujo, False = obstáculo, None = limpo
        self.agent_x = None
        self.agent_y = None
        self.agent_dir = None     # 0=N, 1=E, 2=S, 3=W
        self.total_dirt = 0
        self.cleaned_dirt = 0
        self._reset_grid_and_agent()

    def _reset_grid_and_agent(self):
        self.grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                r = random.random()
                if r < self.obstacle_prob:
                    self.grid[y][x] = False   # obstáculo
                elif r < self.obstacle_prob + self.dirt_prob:
                    self.grid[y][x] = True    # sujo
                else:
                    self.grid[y][x] = None    # limpo

        while True:
            x = random.randint(0, self.width-1)
            y = random.randint(0, self.height-1)
            if self.grid[y][x] is not False:
                self.agent_x = x
                self.agent_y = y
                break
        self.agent_dir = random.randint(0, 3)

        self.total_dirt = sum(cell is True for row in self.grid for cell in row)
        self.cleaned_dirt = 0

    def get_percept(self):
        """Retorna (dirt_current, blocked_front)"""
        dirt = (self.grid[self.agent_y][self.agent_x] is True)
        fx, fy = self._front_cell()
        blocked = (fx < 0 or fx >= self.width or fy < 0 or fy >= self.height or
                   self.grid[fy][fx] is False)
        return (dirt, blocked)

    def _front_cell(self):
        dx, dy = [(0,-1), (1,0), (0,1), (-1,0)][self.agent_dir]
        return (self.agent_x + dx, self.agent_y + dy)

    def execute_action(self, action):
        """
        Executa ação e retorna (percept, reward, done)
        reward conforme métrica: chamada externa decide se penaliza movimento.
        Aqui retornamos +1 para aspiração bem sucedida, e -1 para movimento
        (cabe ao caller ignorar se métrica A). O movimento só acontece se não bloqueado.
        """
        reward = 0
        if action == 'suck':
            if self.grid[self.agent_y][self.agent_x] is True:
                self.grid[self.agent_y][self.agent_x] = None
                self.cleaned_dirt += 1
                reward = 1
        elif action == 'move_forward':
            fx, fy = self._front_cell()
            if (0 <= fx < self.width and 0 <= fy < self.height and
                self.grid[fy][fx] is not False):
                self.agent_x, self.agent_y = fx, fy
                reward = -1   # custo do movimento
        elif action == 'turn_left':
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == 'turn_right':
            self.agent_dir = (self.agent_dir + 1) % 4

        done = (self.cleaned_dirt == self.total_dirt)
        return self.get_percept(), reward, done

    def reset(self):
        """Reinicia o ambiente com nova configuração aleatória."""
        self._reset_grid_and_agent()