"""
Battlesnake AI с Monte Carlo Tree Search и усиленной оценочной функцией.
 
Интегрированы компоненты из эвристического модуля:
- _head_to_head_cells для точного штрафа опасных клеток.
- _flood_fill для проверки на застревание.
- Эвристическая roll-out политика для нашей змеи.
- Одновременное выполнение ходов всех змей (корректная симуляция).
"""
 
from collections import deque
from typing import Dict, List, Optional, Set, Tuple
import copy
import math
import random
import time
 
Point = Tuple[int, int]
 
DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0),
}
 
HEAD_TO_HEAD_PENALTY = 10_000
HUNGRY_THRESHOLD = 50
 
def get_info() -> Dict[str, str]:
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.6.0",
    }
 
# =========================================================================
#  Геометрия и BFS (общие утилиты)
# =========================================================================
def _in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h
 
def _neighbors(x: int, y: int, w: int, h: int):
    for dx, dy in DIRECTIONS.values():
        nx, ny = x + dx, y + dy
        if _in_bounds(nx, ny, w, h):
            yield nx, ny
 
def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
 
def _occupied(snakes: List[Dict]) -> Set[Point]:
    occ = set()
    for s in snakes:
        for seg in s["body"]:
            occ.add((seg["x"], seg["y"]))
    return occ
 
def bfs(start, blocked: Set[Point], w: int, h: int, max_dist: int = None) -> Dict[Point, int]:
    if isinstance(start, tuple):
        starts = [start]
    else:
        starts = start
    dist = {}
    q = deque()
    for s in starts:
        if s not in blocked:
            dist[s] = 0
            q.append(s)
    while q:
        x, y = q.popleft()
        d = dist[(x, y)]
        if max_dist is not None and d >= max_dist:
            continue
        for nx, ny in _neighbors(x, y, w, h):
            if (nx, ny) not in blocked and (nx, ny) not in dist:
                dist[(nx, ny)] = d + 1
                q.append((nx, ny))
    return dist
 
def _flood_fill(start: Point, occupied: Set[Point], width: int, height: int, limit: int) -> int:
    """Подсчёт достижимых клеток (с лимитом) — из эвристического модуля."""
    seen: Set[Point] = {start}
    stack: List[Point] = [start]
    count = 0
    while stack:
        x, y = stack.pop()
        count += 1
        if count >= limit:
            break
        for dx, dy in DIRECTIONS.values():
            nbr = (x + dx, y + dy)
            if nbr in seen:
                continue
            if not _in_bounds(nbr, width, height):
                continue
            if nbr in occupied:
                continue
            seen.add(nbr)
            stack.append(nbr)
    return count
 
def _head_to_head_cells(snakes: List[Dict], my_id: str, my_length: int) -> Set[Point]:
    """Клетки, соседние с головами врагов, которые >= нашей длины."""
    danger: Set[Point] = set()
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        if snake["length"] < my_length:
            continue
        ehead = (snake["head"]["x"], snake["head"]["y"])
        for dx, dy in DIRECTIONS.values():
            danger.add((ehead[0] + dx, ehead[1] + dy))
    return danger
 
# =========================================================================
#  Безопасные ходы (с учётом head-to-head)
# =========================================================================
def safe_moves(state: Dict, snake_id: str = None) -> List[str]:
    if snake_id is None:
        snake_id = state["you"]["id"]
    snake = next(s for s in state["board"]["snakes"] if s["id"] == snake_id)
    head = snake["head"]
    my_len = snake["length"]
    w, h = state["board"]["width"], state["board"]["height"]
    occ = _occupied(state["board"]["snakes"])
 
    enemy_heads = {}
    for s in state["board"]["snakes"]:
        if s["id"] != snake_id:
            enemy_heads[(s["head"]["x"], s["head"]["y"])] = s["length"]
 
    safe = []
    for move, (dx, dy) in DIRECTIONS.items():
        nx, ny = head["x"] + dx, head["y"] + dy
        if not _in_bounds(nx, ny, w, h) or (nx, ny) in occ:
            continue
        opp_len = enemy_heads.get((nx, ny))
        if opp_len is not None and opp_len >= my_len:
            continue
        safe.append(move)
    return safe
 
# =========================================================================
#  Одновременный шаг игры (заменяет _apply_move / _update_board)
# =========================================================================
def step(state: Dict, moves: Dict[str, str]) -> None:
    """
    Применяет ходы всех змей одновременно.
    Изменяет переданное состояние (должно быть копией).
    """
    board = state["board"]
    snakes = board["snakes"]
    w, h = board["width"], board["height"]
 
    # 1. Сохраняем старые хвосты для каждой змеи
    old_tails = {}
    for snake in snakes:
        body = snake["body"]
        if len(body) > 1:
            old_tails[snake["id"]] = (body[-1]["x"], body[-1]["y"])
        else:
            old_tails[snake["id"]] = None
 
    # 2. Двигаем головы, запоминаем новые позиции и флаг роста
    new_heads = {}
    grew = {}
    for snake in snakes:
        sid = snake["id"]
        if sid not in moves:
            continue  # змея не делает ход (такого быть не должно)
        dx, dy = DIRECTIONS[moves[sid]]
        hx, hy = snake["head"]["x"] + dx, snake["head"]["y"] + dy
        new_heads[sid] = (hx, hy)
        # вставляем новую голову во временный список
        snake["body"].insert(0, {"x": hx, "y": hy})
        snake["head"] = {"x": hx, "y": hy}
        # проверяем еду
        ate = False
        for food in board["food"]:
            if food["x"] == hx and food["y"] == hy:
                ate = True
                board["food"].remove(food)
                break
        grew[sid] = ate
 
    # 3. Строим множество занятых клеток для проверки смертей
    # Учитываем, что хвост (последний сегмент) исчезает, если змея не выросла.
    occupied_after_tails = set()
    for snake in snakes:
        sid = snake["id"]
        tail_pos = old_tails.get(sid)
        for seg in snake["body"]:
            pos = (seg["x"], seg["y"])
            # если змея не выросла и это её хвост (последний сегмент), пропускаем
            if not grew.get(sid, False) and pos == tail_pos:
                continue
            occupied_after_tails.add(pos)
 
    # 4. Определяем погибших
    dead_ids = set()
    # головы, сгруппированные по клетке
    head_positions = {}  # pos -> list of (sid, length)
    for sid, pos in new_heads.items():
        if sid in dead_ids:
            continue
        # выход за границы
        if not _in_bounds(pos[0], pos[1], w, h):
            dead_ids.add(sid)
            continue
        # столкновение с телом (после удаления хвостов)
        if pos in occupied_after_tails:
            dead_ids.add(sid)
            continue
        head_positions.setdefault(pos, []).append((sid, next(s["length"] for s in snakes if s["id"] == sid)))
 
    # столкновения голова-в-голову
    for pos, heads in head_positions.items():
        if len(heads) > 1:
            # выживает самая длинная; при равенстве – все умирают
            max_len = max(h[1] for h in heads)
            survivors = [h for h in heads if h[1] > max_len]
            if len(survivors) == 1:
                # все остальные умирают
                for sid, _ in heads:
                    if sid != survivors[0][0]:
                        dead_ids.add(sid)
            else:
                # все умирают
                for sid, _ in heads:
                    dead_ids.add(sid)
 
    # 5. Превращаем погибших змей в еду и удаляем их из списка
    new_food = []
    for snake in snakes[:]:
        if snake["id"] in dead_ids:
            # всё тело становится едой
            for seg in snake["body"]:
                new_food.append({"x": seg["x"], "y": seg["y"]})
            snakes.remove(snake)
 
    board["food"].extend(new_food)
 
    # 6. Удаляем хвосты у выживших, если не было роста
    for snake in snakes:
        if not grew.get(snake["id"], False):
            if len(snake["body"]) > 1:
                snake["body"].pop()
        # обновляем length
        snake["length"] = len(snake["body"])
 
# =========================================================================
#  Оценка состояния (улучшена)
# =========================================================================
def component_size(head: Dict, board: Dict) -> int:
    occ = _occupied(board["snakes"])
    return len(bfs((head["x"], head["y"]), occ, board["width"], board["height"]))
 
def nearest_food(head: Dict, board: Dict) -> Optional[int]:
    occ = _occupied(board["snakes"])
    dist_map = bfs((head["x"], head["y"]), occ, board["width"], board["height"])
    foods = board["food"]
    if not foods:
        return None
    best = None
    for f in foods:
        d = dist_map.get((f["x"], f["y"]))
        if d is not None and (best is None or d < best):
            best = d
    return best
 
def evaluate(state: Dict, weights: Dict[str, float]) -> float:
    you = state["you"]
    head = you["head"]
    board = state["board"]
    w, h = board["width"], board["height"]
    my_len = you["length"]
 
    occ = _occupied(board["snakes"])
    dist_map = bfs((head["x"], head["y"]), occ, w, h)
 
    # 1. Пространство (два компонента)
    comp = len(dist_map)
    space_score = comp / (w * h)
 
    # Дополнительно: flood fill с лимитом my_length+1 (как в эвристике)
    immediate_space = _flood_fill((head["x"], head["y"]), occ, w, h, my_len + 1)
    if immediate_space < my_len + 1:
        space_penalty = (my_len + 1 - immediate_space) * 0.5
        space_score = max(0, space_score - space_penalty)
 
    # 2. Еда
    fd = nearest_food(head, board)
    food_score = 1.0 / (fd + 1) if fd is not None else 0.0
 
    # 3. Безопасность
    opp_heads = []
    for s in board["snakes"]:
        if s["id"] != you["id"]:
            opp_heads.append((s["head"]["x"], s["head"]["y"]))
 
    if opp_heads:
        min_enemy_dist = min((dist_map.get(h, w + h) for h in opp_heads), default=w + h)
        close_threat = sum(1.0 / (dist_map.get(h, w + h) + 1) for h in opp_heads if dist_map.get(h, w + h) <= 2)
        safety_score = 1.0 / (1.0 + close_threat)
        safety_score = 0.7 * safety_score + 0.3 * min(1.0, min_enemy_dist / (w + h))
 
        # Явный штраф за нахождение в клетке, соседней с опасной вражеской головой
        dangerous_cells = _head_to_head_cells(board["snakes"], you["id"], my_len)
        if (head["x"], head["y"]) in dangerous_cells:
            safety_score -= 0.3
            if safety_score < 0:
                safety_score = 0.0
    else:
        safety_score = 1.0
 
    # 4. Агрессия
    aggression = 0.0
    for s in board["snakes"]:
        if s["id"] != you["id"] and s["length"] < my_len:
            d = dist_map.get((s["head"]["x"], s["head"]["y"]), w + h)
            if d <= 2:
                aggression += 1.0 / (d + 1)
    aggression = min(1.0, aggression)
 
    # 5. Длина
    lengths = [s["length"] for s in board["snakes"]]
    max_len = max(lengths) if lengths else 1
    length_score = my_len / max_len
 
    # 6. Центральность
    cx, cy = (w - 1) / 2, (h - 1) / 2
    center_dist = abs(head["x"] - cx) + abs(head["y"] - cy)
    center_score = 1.0 - center_dist / (w + h)
 
    # 7. Штрафы (смягчены)
    free_neighbors = sum(1 for nx, ny in _neighbors(head["x"], head["y"], w, h) if (nx, ny) not in occ)
    escape_penalty = max(0, (3 - free_neighbors) / 3) * 0.1
    corners = [(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)]
    min_corner = min(abs(head["x"] - cx) + abs(head["y"] - cy) for cx, cy in corners)
    corner_penalty = (1.0 - min_corner / (w + h)) * 0.15
    space_score = max(0, space_score - corner_penalty - escape_penalty)
 
    return (weights["space"]    * space_score +
            weights["food"]     * food_score +
            weights["safety"]   * safety_score +
            weights["aggression"] * aggression +
            weights["length"]   * length_score +
            weights["center"]   * center_score)
 
def get_weights(state: Dict) -> Dict[str, float]:
    you = state["you"]
    health = you["health"]
    num_snakes = len(state["board"]["snakes"])
    my_len = you["length"]
    opp_lens = [s["length"] for s in state["board"]["snakes"] if s["id"] != you["id"]]
    max_opp_len = max(opp_lens) if opp_lens else 0
 
    w = {
        "space": 2.0,
        "food": 2.0,
        "safety": 3.0,
        "aggression": 1.0,
        "length": 1.5,
        "center": 0.5,
    }
    if health < 30:
        w["food"] = 4.0
    elif health < 50:
        w["food"] = 3.0
    if num_snakes > 4:
        w["space"] = 3.0
    if my_len > max_opp_len:
        w["aggression"] = 2.0
    return w
 
# =========================================================================
#  Эвристическая политика (из отдельного модуля)
# =========================================================================
def choose_move_heuristic(game_state: Dict) -> str:
    """Резервная эвристика — теперь используется и в roll-out'ах."""
    board = game_state["board"]
    you = game_state["you"]
    width, height = board["width"], board["height"]
    head = (you["head"]["x"], you["head"]["y"])
    my_length = you["length"]
    health = you["health"]
    occupied = _occupied(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]
 
    best_move = None
    best_score = float("-inf")
    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)
        if not _in_bounds(nxt, width, height):
            continue
        if nxt in occupied:
            continue
 
        space = _flood_fill(nxt, occupied, width, height, limit=my_length + 1)
        score = float(space)
        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY
        if foods and health < HUNGRY_THRESHOLD:
            nearest = min(_manhattan(nxt, f) for f in foods)
            score += (width + height - nearest) * 2
        if score > best_score:
            best_score, best_move = score, move
    return best_move or "up"
 
# =========================================================================
#  Противники – всегда эвристика
# =========================================================================
def _opponent_move_heuristic(state: Dict, snake: Dict) -> str:
    """Эвристика для противника."""
    safe = safe_moves(state, snake["id"])
    if not safe:
        return "up"
    head = snake["head"]
    board = state["board"]
    w, h = board["width"], board["height"]
    foods = board["food"]
    occ = _occupied(board["snakes"])
    best_move = safe[0]
    best_score = -float("inf")
    for move in safe:
        dx, dy = DIRECTIONS[move]
        nx, ny = head["x"] + dx, head["y"] + dy
        space = len(bfs((nx, ny), occ, w, h, max_dist=snake["length"] + 1))
        score = space
        if foods and snake["health"] < HUNGRY_THRESHOLD:
            nearest = min(abs(nx - f["x"]) + abs(ny - f["y"]) for f in foods)
            score += (w + h - nearest) * 2
        min_dist_to_wall = min(nx, w - 1 - nx, ny, h - 1 - ny)
        score += min_dist_to_wall * 0.5
        if score > best_score:
            best_score, best_move = score, move
    return best_move
 
def _opponent_policy(state: Dict, snake: Dict) -> str:
    """Противники используют эвристику без случайности."""
    return _opponent_move_heuristic(state, snake)
 
# =========================================================================
#  MCTS с усиленной roll-out политикой (одновременные ходы)
# =========================================================================
class MCTSNode:
    def __init__(self, state: Dict, move: Optional[str] = None, parent=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.children: Dict[str, MCTSNode] = {}
        self.visits = 0
        self.total_score = 0.0
        self.untried_moves = safe_moves(state)
 
    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0
 
    def best_child(self, exploration: float = 1.414) -> 'MCTSNode':
        best = None
        best_uct = -float('inf')
        for child in self.children.values():
            if child.visits == 0:
                uct = float('inf')
            else:
                exploitation = child.total_score / child.visits
                exploration_term = exploration * math.sqrt(math.log(self.visits) / child.visits)
                uct = exploitation + exploration_term
            if uct > best_uct:
                best_uct = uct
                best = child
        return best
 
def simulate_rollout(state: Dict, max_depth: int = 30) -> float:
    """
    Усиленная roll-out симуляция с одновременными ходами.
    Наша змея ходит по эвристике choose_move_heuristic.
    Противники ходят по _opponent_policy.
    """
    sim = copy.deepcopy(state)
    for _ in range(max_depth):
        if sim["you"]["id"] not in [s["id"] for s in sim["board"]["snakes"]]:
            return -1.0
 
        # Собираем ходы для всех змей
        moves = {}
        my_id = sim["you"]["id"]
        # Наш ход
        my_move = choose_move_heuristic(sim)
        my_safe = safe_moves(sim, my_id)
        if my_move not in my_safe:
            if not my_safe:
                return -1.0
            my_move = random.choice(my_safe)
        moves[my_id] = my_move
 
        # Ходы противников
        for snake in sim["board"]["snakes"]:
            if snake["id"] == my_id:
                continue
            moves[snake["id"]] = _opponent_policy(sim, snake)
 
        # Одновременный шаг
        step(sim, moves)
 
    return evaluate(sim, get_weights(sim))
 
def mcts_choose_move(root_state: Dict, time_limit: float = 0.45) -> str:
    root = MCTSNode(root_state)
    end_time = time.time() + time_limit
 
    while time.time() < end_time:
        node = root
        # 1. Selection
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        # 2. Expansion
        if not node.is_fully_expanded() and node.untried_moves:
            move = random.choice(node.untried_moves)
            node.untried_moves.remove(move)
            new_state = copy.deepcopy(node.state)
 
            # Собираем ходы для всех змей
            moves = {}
            my_id = new_state["you"]["id"]
            moves[my_id] = move
            for snake in new_state["board"]["snakes"]:
                if snake["id"] == my_id:
                    continue
                moves[snake["id"]] = _opponent_policy(new_state, snake)
 
            # Применяем одновременный шаг
            step(new_state, moves)
 
            child = MCTSNode(new_state, move, parent=node)
            node.children[move] = child
            node = child
        # 3. Simulation
        score = simulate_rollout(node.state)
        # 4. Backpropagation
        while node is not None:
            node.visits += 1
            node.total_score += score
            node = node.parent
 
    best_move = None
    best_visits = -1
    for move, child in root.children.items():
        if child.visits > best_visits:
            best_visits = child.visits
            best_move = move
    return best_move
 
# =========================================================================
#  Главный выбор хода
# =========================================================================
def choose_move(game_state: Dict) -> str:
    try:
        safe = safe_moves(game_state)
        if not safe:
            return "up"
        if len(safe) == 1:
            return safe[0]
 
        move = mcts_choose_move(game_state, time_limit=0.45)
        if move is not None:
            return move
    except Exception:
        pass
 
    # Fallback – эвристика
    return choose_move_heuristic(game_state)
