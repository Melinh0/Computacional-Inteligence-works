from collections import deque

class ModelBasedAgent:
    def __init__(self):
        self.name = "ModelBased"
        self.reset()

    def reset(self):
        self.map = {}          # (x,y): True=sujo, False=obstáculo, 'clean'=limpo, None=desconhecido
        self.est_x = 0
        self.est_y = 0
        self.est_dir = 0       # 0=N,1=E,2=S,3=W
        self.frontier = deque()
        self.visited = set()
        self.visited.add((0,0))
        self.map[(0,0)] = None
        self.last_action = None
        self.last_success = False

    def _update_map(self, percept, action, move_success):
        dirt, _ = percept
        x, y = self.est_x, self.est_y

        if dirt:
            self.map[(x,y)] = True
        else:
            self.map[(x,y)] = 'clean'

        if action == 'move_forward':
            dx, dy = [(0,-1), (1,0), (0,1), (-1,0)][self.est_dir]
            fx, fy = x+dx, y+dy
            if move_success:
                self.est_x, self.est_y = fx, fy
                self.visited.add((fx, fy))
                if (fx, fy) not in self.map:
                    self.map[(fx, fy)] = None
            else:
                self.map[(fx, fy)] = False
        elif action == 'turn_left':
            self.est_dir = (self.est_dir - 1) % 4
        elif action == 'turn_right':
            self.est_dir = (self.est_dir + 1) % 4

        for d in range(4):
            dx, dy = [(0,-1), (1,0), (0,1), (-1,0)][d]
            nx, ny = self.est_x + dx, self.est_y + dy
            if (nx, ny) not in self.visited and (nx, ny) not in self.frontier:
                if self.map.get((nx, ny)) is not False:
                    self.frontier.append((nx, ny))

    def _plan_path(self, target):
        start = (self.est_x, self.est_y)
        if start == target:
            return []
        queue = deque([(start, [])])
        visited = {start}
        while queue:
            (cx, cy), path = queue.popleft()
            for d in range(4):
                dx, dy = [(0,-1), (1,0), (0,1), (-1,0)][d]
                nx, ny = cx+dx, cy+dy
                if (nx, ny) == target:
                    actions = path + [self._dir_to_action(d)]
                    return actions
                if (nx, ny) not in visited and self.map.get((nx, ny)) is not False:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [self._dir_to_action(d)]))
        return None

    def _dir_to_action(self, target_dir):
        diff = (target_dir - self.est_dir) % 4
        if diff == 0:
            return 'move_forward'
        elif diff == 1:
            return 'turn_right'
        else:  # 2 ou 3
            return 'turn_left'   # vira à esquerda (pode ser usado para 180° também)

    def choose_action(self, percept):
        dirt, _ = percept
        if dirt:
            return 'suck'

        dirty_cells = [pos for pos, state in self.map.items() if state is True]
        if dirty_cells:
            target = min(dirty_cells, key=lambda p: abs(p[0]-self.est_x)+abs(p[1]-self.est_y))
            path = self._plan_path(target)
            if path:
                return path[0]

        while self.frontier:
            target = self.frontier.popleft()
            if self.map.get(target) is False:
                continue
            path = self._plan_path(target)
            if path:
                self.frontier.appendleft(target)
                return path[0]
        return 'shut_off'

    def update_after_action(self, percept, action, move_success):
        self._update_map(percept, action, move_success)