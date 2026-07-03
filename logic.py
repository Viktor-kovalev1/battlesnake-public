"""
Лаконичный модуль логики Battlesnake с фокусом на рост и выживание.
 
- Убран жёсткий порог голода (еда всегда ценна).
- Добавлен стимул к длине (length_score).
- Смягчены штрафы за углы/тупики.
- Упрощена архитектура (общая BFS, нет модели ML).
- Противники в симуляции двигаются разумно, но предсказуемо.
"""
 
from collections import deque
from typing import Dict, List, Optional, Set, Tuple
import copy
import time
 
Point = Tuple[int, int]
 
DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0),
}
 
def get_info() -> Dict[str, str]:
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.2.0",
    }
 
# ---------------------------------------------------------------------------
#  Геометрия и BFS (единая функция)
# ---------------------------------------------------------------------------
def _in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h
 
def _neighbors(x: int, y: int, w: int, h: int):
    for dx, dy in DIRECTIONS.values():
        nx, ny = x + dx, y + dy
        if _in_bounds(nx, ny, w, h):
            yield nx, ny
 
def _occupied(snakes: List[Dict]) -> Set[Point]:
    occ = set()
    for s in snakes:
        for seg in s["body"]:
            occ.add((seg["x"], seg["y"]))
    return occ
 
def bfs(start, blocked: Set[Point], w: int, h: int, max_dist: int = None) -> Dict[Point, int]:
    """Универсальный BFS: от одной или нескольких стартовых клеток."""
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
 
# ---------------------------------------------------------------------------
#  Безопасные ходы
# ---------------------------------------------------------------------------
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
 
# ---------------------------------------------------------------------------
#  Оценка состояния (исправленные веса и новые компоненты)
# ---------------------------------------------------------------------------
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
 
    # 1. Пространство
    comp = len(dist_map)
    space_score = comp / (w * h)
 
    # 2. Еда
    fd = nearest_food(head, board)
    food_score = 1.0 / (fd + 1) if fd is not None else 0.0
 
    # 3. Безопасность (расстояние до вражеских голов)
    opp_heads = []
    for s in board["snakes"]:
        if s["id"] != you["id"]:
            opp_heads.append((s["head"]["x"], s["head"]["y"]))
    if opp_heads:
        min_enemy_dist = min((dist_map.get(h, w + h) for h in opp_heads), default=w + h)
    else:
        min_enemy_dist = w + h
    safety_score = min(1.0, min_enemy_dist / (w + h))
 
    # 4. Агрессия (можем съесть короткого врага)
    aggression = 0.0
    for s in board["snakes"]:
        if s["id"] != you["id"] and s["length"] < my_len:
            d = dist_map.get((s["head"]["x"], s["head"]["y"]), w + h)
            if d <= 2:
                aggression += 1.0 / (d + 1)
    aggression = min(1.0, aggression)
 
    # 5. Стимул к длине
    lengths = [s["length"] for s in board["snakes"]]
    max_len = max(lengths) if lengths else 1
    length_score = my_len / max_len
 
    # 6. Центральность
    cx, cy = (w - 1) / 2, (h - 1) / 2
    center_dist = abs(head["x"] - cx) + abs(head["y"] - cy)
    center_score = 1.0 - center_dist / (w + h)
 
    # 7. Штрафы (смягчены)
    free_neighbors = sum(1 for nx, ny in _neighbors(head["x"], head["y"], w, h) if (nx, ny) not in occ)
    escape_penalty = max(0, (3 - free_neighbors) / 3) * 0.1   # было 0.2
    corners = [(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)]
    min_corner = min(abs(head["x"] - cx) + abs(head["y"] - cy) for cx, cy in corners)
    corner_penalty = (1.0 - min_corner / (w + h)) * 0.15       # было 0.3
    space_score = max(0, space_score - corner_penalty - escape_penalty)
 
    return (weights["space"]   * space_score +
            weights["food"]    * food_score +
            weights["safety"]  * safety_score +
            weights["aggression"] * aggression +
            weights["length"]  * length_score +
            weights["center"]  * center_score)
 
def get_weights(state: Dict) -> Dict[str, float]:
    you = state["you"]
    health = you["health"]
    num_snakes = len(state["board"]["snakes"])
    my_len = you["length"]
    opp_lens = [s["length"] for s in state["board"]["snakes"] if s["id"] != you["id"]]
    max_opp_len = max(opp_lens) if opp_lens else 0
 
    w = {
        "space": 2.0,
        "food": 2.0,          # еда теперь всегда важна
        "safety": 2.0,        # снижено с 4.0
        "aggression": 1.0,
        "length": 1.5,        # новый вес
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
 
# ---------------------------------------------------------------------------
#  Симуляция
# ---------------------------------------------------------------------------
def _apply_move(state: Dict, snake_id: str, move: str) -> None:
    snake = next(s for s in state["board"]["snakes"] if s["id"] == snake_id)
    dx, dy = DIRECTIONS[move]
    new_head = {"x": snake["head"]["x"] + dx, "y": snake["head"]["y"] + dy}
    snake["body"].insert(0, new_head)
    snake["head"] = new_head
 
def _update_board(state: Dict) -> None:
    board = state["board"]
    foods = board["food"]
    eaten = set()
    for s in board["snakes"]:
        h = s["head"]
        for f in foods:
            if f["x"] == h["x"] and f["y"] == h["y"]:
                eaten.add((f["x"], f["y"]))
    board["food"] = [f for f in foods if (f["x"], f["y"]) not in eaten]
    for s in board["snakes"]:
        if (s["head"]["x"], s["head"]["y"]) in eaten:
            s["length"] += 1
        else:
            if len(s["body"]) > 1:
                s["body"].pop()
        s["length"] = len(s["body"])
 
def _opponent_move(state: Dict, snake: Dict) -> str:
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
        if foods and snake["health"] < 50:
            nearest = min(abs(nx - f["x"]) + abs(ny - f["y"]) for f in foods)
            score += (w + h - nearest) * 2
        if score > best_score:
            best_score, best_move = score, move
    return best_move
 
def simulate(state: Dict, move: str, depth: int) -> Dict:
    sim = copy.deepcopy(state)
    _apply_move(sim, sim["you"]["id"], move)
    for _ in range(depth - 1):
        for snake in sim["board"]["snakes"]:
            if snake["id"] == sim["you"]["id"]:
                continue
            _apply_move(sim, snake["id"], _opponent_move(sim, snake))
        _update_board(sim)
    return sim
 
# ---------------------------------------------------------------------------
#  Главный выбор хода
# ---------------------------------------------------------------------------
def choose_move(game_state: Dict) -> str:
    start = time.time()
    safe = safe_moves(game_state)
    if not safe:
        return "up"
    if len(safe) == 1:
        return safe[0]
 
    weights = get_weights(game_state)
    head = game_state["you"]["head"]
    board = game_state["board"]
    w, h = board["width"], board["height"]
    space_ratio = component_size(head, board) / (w * h)
    num_snakes = len(board["snakes"])
 
    # Адаптивная глубина симуляции
    if space_ratio < 0.2 and num_snakes <= 3:
        depth = 3
    elif space_ratio < 0.3 or num_snakes <= 4:
        depth = 2
    else:
        depth = 1
    depth = min(depth, 2)  # ограничение для скорости
 
    best_move = safe[0]
    best_score = -float("inf")
    for move in safe:
        sim_state = simulate(game_state, move, depth)
        score = evaluate(sim_state, weights)
        if score > best_score:
            best_score, best_move = score, move
        if time.time() - start > 0.45:
            break
    return best_move
