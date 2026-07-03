"""
Модуль логики выбора хода для змеи Battlesnake.
 
Реализует гибридный алгоритм принятия решений с симуляцией, динамическими
весами и улучшенной оценкой пространства для предотвращения застревания.
 
Координаты доски: (0, 0) — нижний левый угол.
  up    -> y + 1
  down  -> y - 1
  left  -> x - 1
  right -> x + 1
 
Схема игрового состояния: https://docs.battlesnake.com/api
"""
 
from collections import deque
from typing import Dict, List, Optional, Set, Tuple
import copy
import time
import math
 
# Тип для точки на доске (x, y)
Point = Tuple[int, int]
 
# Сопоставление названий направлений с их векторами на доске
DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}
 
# Штраф за ход, который может привести к проигрышному столкновению головами
HEAD_TO_HEAD_PENALTY = 10_000
# Порог здоровья, ниже которого змея начинает активно искать еду
HUNGRY_THRESHOLD = 50
 
def get_info() -> Dict[str, str]:
    """
    Возвращает метаданные о змее для эндпоинта GET /.
 
    Returns:
        Dict[str, str]: Словарь с версией API, цветом, стилем головы и хвоста.
    """
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.1.0",
    }
 
# ======================================================================
#  ОРИГИНАЛЬНЫЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (сохранены без изменений)
# ======================================================================
 
def _occupied_cells(snakes: List[Dict]) -> Set[Point]:
    """
    Возвращает множество клеток, занятых телами всех змей.
 
    Хвосты также считаются занятыми — это консервативный подход,
    поскольку хвост освобождается только на следующем ходу.
 
    Args:
        snakes (List[Dict]): Список змей из игрового состояния.
 
    Returns:
        Set[Point]: Множество координат занятых клеток.
    """
    occupied: Set[Point] = set()
    for snake in snakes:
        for seg in snake["body"]:
            occupied.add((seg["x"], seg["y"]))
    return occupied
 
def _head_to_head_cells(snakes: List[Dict], my_id: str, my_length: int) -> Set[Point]:
    """
    Возвращает клетки, соседние с головами врагов, которые имеют длину >= нашей.
 
    Перемещение в такую клетку может привести к столкновению головами,
    которое мы проиграем или сведём вничью. Такие ходы сильно штрафуются,
    но не запрещаются — иногда это единственный выход.
 
    Args:
        snakes (List[Dict]): Список змей.
        my_id (str): Идентификатор нашей змеи.
        my_length (int): Длина нашей змеи.
 
    Returns:
        Set[Point]: Множество опасных клеток.
    """
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
 
def _flood_fill(start: Point, occupied: Set[Point], width: int, height: int, limit: int) -> int:
    """
    Подсчитывает количество свободных клеток, достижимых из start, но не более limit.
 
    Используется для оценки, не запрёт ли нас выбранный ход в маленьком кармане.
 
    Args:
        start (Point): Стартовая клетка.
        occupied (Set[Point]): Множество занятых клеток.
        width (int): Ширина доски.
        height (int): Высота доски.
        limit (int): Максимальное количество клеток для подсчёта.
 
    Returns:
        int: Количество достижимых клеток (обрывается на limit).
    """
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
 
def _in_bounds(p: Point, width: int, height: int) -> bool:
    """
    Проверяет, находится ли точка в пределах доски.
 
    Args:
        p (Point): Координаты точки.
        width (int): Ширина доски.
        height (int): Высота доски.
 
    Returns:
        bool: True, если точка внутри доски.
    """
    return 0 <= p[0] < width and 0 <= p[1] < height
 
def _manhattan(a: Point, b: Point) -> int:
    """
    Вычисляет манхэттенское расстояние между двумя точками.
 
    Args:
        a (Point): Первая точка.
        b (Point): Вторая точка.
 
    Returns:
        int: Сумма модулей разностей координат.
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
 
# ======================================================================
#  ФУНКЦИИ ДЛЯ МОДЕЛИ МАШИННОГО ОБУЧЕНИЯ (оригинальные)
# ======================================================================
 
_BIG = 10_000
_NEIGHBORS = ((0, 1), (0, -1), (-1, 0), (1, 0))
 
def _bfs_dist(sources, blocked, width, height):
    """
    Выполняет поиск в ширину от нескольких источников, игнорируя заблокированные клетки.
 
    Args:
        sources (Iterable[Point]): Начальные клетки.
        blocked (Set[Point]): Множество непроходимых клеток.
        width (int): Ширина доски.
        height (int): Высота доски.
 
    Returns:
        Dict[Point, int]: Словарь расстояний от ближайшего источника до каждой достижимой клетки.
    """
    dist = {}
    dq = deque()
    for source in sources:
        if source not in dist:
            dist[source] = 0
            dq.append(source)
    while dq:
        x, y = dq.popleft()
        d = dist[(x, y)]
        for dx, dy in _NEIGHBORS:
            nb = (x + dx, y + dy)
            if 0 <= nb[0] < width and 0 <= nb[1] < height and nb not in blocked and nb not in dist:
                dist[nb] = d + 1
                dq.append(nb)
    return dist
 
def _candidate_features(state: Dict, move: str) -> Dict[str, float]:
    """
    Вычисляет вектор признаков для заданного хода (предполагается, что ход легален).
 
    Используется для подачи в линейную модель.
 
    Args:
        state (Dict): Игровое состояние.
        move (str): Название направления.
 
    Returns:
        Dict[str, float]: Словарь признаков.
    """
    board = state["board"]
    you = state["you"]
    width, height = board["width"], board["height"]
    head = (you["head"]["x"], you["head"]["y"])
    my_length = you["length"]
    health = you["health"]
 
    dx, dy = DIRECTIONS[move]
    nxt = (head[0] + dx, head[1] + dy)
 
    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]
    enemies = [s for s in board["snakes"] if s["id"] != you["id"]]
    enemy_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies]
    bigger_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies if s["length"] >= my_length]
 
    # Воронка: клетки, до которых мы добираемся быстрее любого врага
    my_dist = _bfs_dist([nxt], occupied, width, height)
    enemy_dist = _bfs_dist(enemy_heads, occupied, width, height) if enemy_heads else {}
    voronoi = sum(1 for cell, md in my_dist.items() if md < enemy_dist.get(cell, _BIG))
 
    # Достижимость хвоста (полезно для избегания самоблокировки)
    my_tail = (you["body"][-1]["x"], you["body"][-1]["y"])
    reach = _bfs_dist([nxt], occupied - {my_tail}, width, height)
    reaches_tail = 1.0 if my_tail in reach else 0.0
 
    # Количество свободных соседей у целевой клетки
    escape = sum(
        1
        for ddx, ddy in _NEIGHBORS
        if _in_bounds((nxt[0] + ddx, nxt[1] + ddy), width, height)
        and (nxt[0] + ddx, nxt[1] + ddy) not in occupied
    )
 
    nearest_now = min((_manhattan(head, f) for f in foods), default=_BIG)
    nearest_next = min((_manhattan(nxt, f) for f in foods), default=_BIG)
    hungry = health < HUNGRY_THRESHOLD
 
    return {
        "space_capped": float(_flood_fill(nxt, occupied, width, height, limit=my_length + 1)),
        "open_space": float(_flood_fill(nxt, occupied, width, height, limit=width * height)),
        "voronoi": float(voronoi),
        "reaches_tail": reaches_tail,
        "escape": float(escape),
        "h2h_danger": 1.0 if nxt in danger else 0.0,
        "near_bigger_head": float(min((_manhattan(nxt, h) for h in bigger_heads), default=width + height)),
        "near_enemy_head": float(min((_manhattan(nxt, h) for h in enemy_heads), default=width + height)),
        "wall_dist": float(min(nxt[0], width - 1 - nxt[0], nxt[1], height - 1 - nxt[1])),
        "food_score": float((width + height - nearest_next) * 2) if hungry and foods else 0.0,
        "food_delta": float(nearest_now - nearest_next) if foods else 0.0,
        "is_food": 1.0 if nxt in foods else 0.0,
        "dist_to_center": abs(nxt[0] - (width - 1) / 2) + abs(nxt[1] - (height - 1) / 2),
    }
 
# Встроенная линейная модель (обучена заранее, веса встроены в код)
_MODEL: Dict = {
    "feature_names": [
        "space_capped",
        "open_space",
        "voronoi",
        "reaches_tail",
        "escape",
        "h2h_danger",
        "near_bigger_head",
        "near_enemy_head",
        "wall_dist",
        "food_score",
        "food_delta",
        "is_food",
        "dist_to_center",
    ],
    "mean": [
        7.357954545454546,
        100.9034090909091,
        48.26988636363637,
        0.9943181818181818,
        2.4431818181818183,
        0.04261363636363636,
        9.673295454545455,
        4.676136363636363,
        1.625,
        0.8920454545454546,
        0.14772727272727273,
        0.036931818181818184,
        5.056818181818182,
    ],
    "std": [
        3.5995966185276513,
        22.80542174802676,
        31.41119158524981,
        0.07516338951888041,
        0.6235520417417705,
        0.20198444088469822,
        7.9675173248507924,
        2.2532045017839604,
        1.3552297691803878,
        5.861056404757769,
        0.9449599886584031,
        0.18859442989548575,
        2.34451950177747,
    ],
    "coef": [
        0.00010539398521136327,
        -1.6778512168946185,
        80.89420182766183,
        9.793855564450467,
        0.7884630868036275,
        -11.025170822665032,
        -0.7981723553489,
        0.5410534990053248,
        1.5629078731518526,
        7.582325762611304,
        0.12463070008097832,
        0.21036618806863483,
        1.836259515524985,
    ],
    "intercept": 0.0,
    "top1_accuracy": 0.9928571428571429,
}
 
def choose_move_model(game_state: Dict) -> Optional[str]:
    """
    Выбирает ход с помощью линейной модели.
 
    Если модель недоступна или нет легальных ходов, возвращает None,
    чтобы вызвать эвристический fallback.
 
    Args:
        game_state (Dict): Игровое состояние.
 
    Returns:
        Optional[str]: Лучшее направление или None.
    """
    legal = _legal_moves(game_state)
    if not legal:
        return None
 
    names = _MODEL["feature_names"]
    mean = _MODEL["mean"]
    std = _MODEL["std"]
    coef = _MODEL["coef"]
    intercept = _MODEL["intercept"]
 
    best_move, best_score = None, float("-inf")
    for move in legal:
        feats = _candidate_features(game_state, move)
        score = intercept
        for i, name in enumerate(names):
            z = (feats.get(name, 0.0) - mean[i]) / std[i] if std[i] else 0.0
            score += coef[i] * z
        if score > best_score:
            best_score, best_move = score, move
    return best_move
 
def _legal_moves(game_state: Dict) -> List[str]:
    """
    Возвращает список легальных ходов (не ведущих в стену или чужое тело).
 
    Args:
        game_state (Dict): Игровое состояние.
 
    Returns:
        List[str]: Список допустимых направлений.
    """
    board = game_state["board"]
    width, height = board["width"], board["height"]
    head = (game_state["you"]["head"]["x"], game_state["you"]["head"]["y"])
    occupied = _occupied_cells(board["snakes"])
    return [
        move
        for move, (dx, dy) in DIRECTIONS.items()
        if _in_bounds((head[0] + dx, head[1] + dy), width, height)
        and (head[0] + dx, head[1] + dy) not in occupied
    ]
 
# ======================================================================
#  СТАРЫЙ ГЛАВНЫЙ МЕТОД (используется как fallback)
# ======================================================================
 
def choose_move_original(game_state: Dict) -> str:
    """
    Оригинальная логика выбора хода (модель + эвристика) — используется как резервная.
 
    Args:
        game_state (Dict): Игровое состояние.
 
    Returns:
        str: Направление хода.
    """
    try:
        move = choose_move_model(game_state)
    except Exception:  # noqa: BLE001 - ошибка модели не должна ломать игру
        move = None
    if move is not None:
        return move
    return choose_move_heuristic(game_state)
 
def choose_move_heuristic(game_state: Dict) -> str:
    """
    Чисто эвристический выбор хода (без модели) — гарантирует легальность.
 
    Args:
        game_state (Dict): Игровое состояние.
 
    Returns:
        str: Направление хода.
    """
    board = game_state["board"]
    you = game_state["you"]
    width: int = board["width"]
    height: int = board["height"]
 
    head: Point = (you["head"]["x"], you["head"]["y"])
    my_length: int = you["length"]
    health: int = you["health"]
 
    occupied = _occupied_cells(board["snakes"])
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
 
        # Подсчёт достижимого пространства (если места меньше длины тела — мы запрём себя)
        space = _flood_fill(nxt, occupied, width, height, limit=my_length + 1)
        score = float(space)
 
        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY
 
        # Если голодны — подталкиваем к еде
        if foods and health < HUNGRY_THRESHOLD:
            nearest = min(_manhattan(nxt, f) for f in foods)
            score += (width + height - nearest) * 2
 
        if score > best_score:
            best_score = score
            best_move = move
 
    # Если безопасного хода нет — едем вверх (аварийный вариант)
    return best_move or "up"
 
# ======================================================================
#  НОВЫЙ УЛУЧШЕННЫЙ АЛГОРИТМ (симуляция + динамические веса + умные враги)
# ======================================================================
 
def get_safe_moves(state: Dict, snake_id: Optional[str] = None) -> List[str]:
    """
    Возвращает список ходов, которые не ведут к немедленной гибели для указанной змеи.
 
    Проверяет:
        - выход за пределы доски,
        - столкновение с телами змей,
        - проигрышное столкновение головами (с более длинным или равным врагом).
 
    Если snake_id не указан, используется наша змея (state['you']['id']).
 
    Args:
        state (Dict): Игровое состояние.
        snake_id (Optional[str]): Идентификатор змеи. Если None, берётся наша.
 
    Returns:
        List[str]: Список безопасных направлений.
    """
    if snake_id is None:
        snake_id = state['you']['id']
 
    # Находим змею по id
    snake = None
    for s in state['board']['snakes']:
        if s['id'] == snake_id:
            snake = s
            break
    if snake is None:
        return []
 
    my_head = snake['head']
    my_length = snake['length']
    board = state['board']
    width, height = board['width'], board['height']
 
    occupied = set()
    for s in board['snakes']:
        for seg in s['body']:
            occupied.add((seg['x'], seg['y']))
 
    # Сопоставляем голову врага с его длиной
    head_to_length = {}
    for s in board['snakes']:
        h = s['head']
        head_to_length[(h['x'], h['y'])] = s['length']
 
    safe = []
    for move, (dx, dy) in DIRECTIONS.items():
        nx = my_head['x'] + dx
        ny = my_head['y'] + dy
 
        if not (0 <= nx < width and 0 <= ny < height):
            continue
        if (nx, ny) in occupied:
            continue
 
        opp_len = head_to_length.get((nx, ny))
        if opp_len is not None and opp_len >= my_length:
            continue
 
        safe.append(move)
    return safe
 
def flood_fill_space(head: Dict, board: Dict, max_steps: int) -> int:
    """
    Подсчитывает количество клеток, достижимых из головы за max_steps шагов,
    не проходя через тела змей.
 
    Используется для оценки доступного пространства.
 
    Args:
        head (Dict): Координаты головы {'x': int, 'y': int}.
        board (Dict): Игровая доска.
        max_steps (int): Максимальное количество шагов.
 
    Returns:
        int: Количество достижимых клеток.
    """
    occupied = set()
    for snake in board['snakes']:
        for seg in snake['body']:
            occupied.add((seg['x'], seg['y']))
 
    visited = set()
    q = deque()
    q.append((head['x'], head['y'], 0))
    visited.add((head['x'], head['y']))
    count = 0
 
    while q:
        x, y, dist = q.popleft()
        if dist > max_steps:
            continue
        count += 1
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < board['width'] and 0 <= ny < board['height']:
                if (nx, ny) not in visited and (nx, ny) not in occupied:
                    visited.add((nx, ny))
                    q.append((nx, ny, dist + 1))
    return count
 
def component_size(head: Dict, board: Dict) -> int:
    """
    Возвращает размер связной области (все достижимые клетки),
    игнорируя тела змей как препятствия. Не ограничивает глубину.
 
    Args:
        head (Dict): Координаты головы.
        board (Dict): Игровая доска.
 
    Returns:
        int: Размер компоненты.
    """
    occupied = set()
    for snake in board['snakes']:
        for seg in snake['body']:
            occupied.add((seg['x'], seg['y']))
 
    visited = set()
    stack = [(head['x'], head['y'])]
    visited.add((head['x'], head['y']))
    count = 0
 
    while stack:
        x, y = stack.pop()
        count += 1
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < board['width'] and 0 <= ny < board['height']:
                if (nx, ny) not in visited and (nx, ny) not in occupied:
                    visited.add((nx, ny))
                    stack.append((nx, ny))
    return count
 
def min_distance_to_food(head: Dict, board: Dict) -> Optional[int]:
    """
    Вычисляет кратчайшее расстояние от головы до любой еды (BFS),
    игнорируя тела змей как препятствия.
 
    Args:
        head (Dict): Координаты головы.
        board (Dict): Игровая доска.
 
    Returns:
        Optional[int]: Расстояние до ближайшей еды или None, если еда недостижима.
    """
    occupied = set()
    for snake in board['snakes']:
        for seg in snake['body']:
            occupied.add((seg['x'], seg['y']))
 
    foods = board['food']
    if not foods:
        return None
 
    q = deque()
    q.append((head['x'], head['y'], 0))
    visited = {(head['x'], head['y'])}
 
    while q:
        x, y, dist = q.popleft()
        if any(f['x'] == x and f['y'] == y for f in foods):
            return dist
        for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < board['width'] and 0 <= ny < board['height']:
                if (nx, ny) not in visited and (nx, ny) not in occupied:
                    visited.add((nx, ny))
                    q.append((nx, ny, dist + 1))
    return None
 
def compute_risk(state: Dict) -> float:
    """
    Оценивает риск столкновения с врагами в ближайшее время.
 
    Возвращает число от 0 до 1, где 1 — максимальный риск.
    Учитываются вражеские головы на расстоянии <= 2 клеток.
 
    Args:
        state (Dict): Игровое состояние.
 
    Returns:
        float: Значение риска.
    """
    my_head = state['you']['head']
    board = state['board']
    risk = 0.0
    total = 0
    for snake in board['snakes']:
        if snake['id'] == state['you']['id']:
            continue
        total += 1
        opp_head = snake['head']
        dist = abs(opp_head['x'] - my_head['x']) + abs(opp_head['y'] - my_head['y'])
        if dist <= 2:
            risk += 1.0 / (dist + 1)
    if total == 0:
        return 0.0
    return min(risk / total, 1.0)
 
def compute_aggression(state: Dict) -> float:
    """
    Вычисляет бонус за возможность съесть более короткого врага.
 
    Возвращает значение от 0 до 1, где больше — выгоднее.
 
    Args:
        state (Dict): Игровое состояние.
 
    Returns:
        float: Агрессивный бонус.
    """
    my_head = state['you']['head']
    my_length = state['you']['length']
    board = state['board']
    bonus = 0.0
    for snake in board['snakes']:
        if snake['id'] == state['you']['id']:
            continue
        if snake['length'] < my_length:
            opp_head = snake['head']
            dist = abs(opp_head['x'] - my_head['x']) + abs(opp_head['y'] - my_head['y'])
            if dist <= 2:
                bonus += 1.0 / (dist + 1)
    return min(bonus, 1.0)
 
def get_dynamic_weights(state: Dict) -> Dict[str, float]:
    """
    Возвращает словарь весов для критериев оценки состояния,
    адаптированных под текущую игровую ситуацию.
 
    Веса зависят от здоровья, числа змей и относительной длины.
 
    Args:
        state (Dict): Игровое состояние.
 
    Returns:
        Dict[str, float]: Словарь весов для критериев:
            'space', 'food', 'risk', 'aggression', 'center'.
    """
    health = state['you']['health']
    num_snakes = len(state['board']['snakes'])
    my_length = state['you']['length']
    max_opp_len = 0
    for s in state['board']['snakes']:
        if s['id'] != state['you']['id']:
            max_opp_len = max(max_opp_len, s['length'])
 
    weights = {
        'space': 2.0,
        'food': 0.5,
        'risk': 3.0,
        'aggression': 0.5,
        'center': 0.3
    }
 
    # Если здоровье низкое — еда становится приоритетом
    if health < 40:
        weights['food'] = 1.5 + (40 - health) / 20.0
    else:
        weights['food'] = 0.5
 
    # Если мы самые длинные и змей мало — можно агрессировать
    if num_snakes <= 3 and my_length > max_opp_len:
        weights['aggression'] = 2.0
 
    # При большом количестве змей важнее контролировать пространство
    if num_snakes > 4:
        weights['space'] = 3.0
 
    # Безопасность всегда в приоритете
    weights['risk'] = 4.0
 
    return weights
 
def apply_move_to_snake(state: Dict, snake_id: str, move: str) -> Dict:
    """
    Применяет ход к конкретной змее в состоянии (изменяет переданный словарь).
 
    Вставляет новую голову в начало тела, не удаляя хвост (это делается позже).
 
    Args:
        state (Dict): Игровое состояние (будет изменено).
        snake_id (str): Идентификатор змеи.
        move (str): Направление хода.
 
    Returns:
        Dict: Изменённое состояние (для удобства).
    """
    snake = None
    for s in state['board']['snakes']:
        if s['id'] == snake_id:
            snake = s
            break
    if snake is None:
        return state
 
    head = snake['head']
    dx, dy = DIRECTIONS[move]
    new_head = {'x': head['x'] + dx, 'y': head['y'] + dy}
 
    snake['body'].insert(0, new_head)
    snake['head'] = new_head
    return state
 
def choose_opponent_move_smart(state: Dict, snake: Dict) -> str:
    """
    Умная эвристика для выбора хода вражеской змеи в симуляции.
 
    Использует те же безопасные ходы и оценку по пространству и еде,
    но без рекурсивной симуляции.
 
    Args:
        state (Dict): Игровое состояние.
        snake (Dict): Данные вражеской змеи.
 
    Returns:
        str: Направление хода.
    """
    safe = get_safe_moves(state, snake['id'])
    if not safe:
        return 'up'
 
    board = state['board']
    width, height = board['width'], board['height']
    head = snake['head']
    my_length = snake['length']
    health = snake['health']
    foods = board['food']
 
    best_move = None
    best_score = -float('inf')
 
    for move in safe:
        dx, dy = DIRECTIONS[move]
        nxt = (head['x'] + dx, head['y'] + dy)
 
        occupied = set()
        for s in board['snakes']:
            for seg in s['body']:
                occupied.add((seg['x'], seg['y']))
 
        # Пространство после хода
        space = _flood_fill(nxt, occupied, width, height, limit=my_length + 1)
        score = float(space)
 
        # Если голоден, поощряем движение к еде
        if foods and health < HUNGRY_THRESHOLD:
            nearest = min(_manhattan(nxt, f) for f in foods)
            score += (width + height - nearest) * 2
 
        # Избегаем стен (штраф за близость к углу)
        min_dist_to_wall = min(nxt[0], width - 1 - nxt[0], nxt[1], height - 1 - nxt[1])
        score += min_dist_to_wall * 0.5
 
        if score > best_score:
            best_score = score
            best_move = move
 
    return best_move or 'up'
 
def update_after_moves(state: Dict) -> Dict:
    """
    Обновляет состояние после того, как все змеи сделали ход:
        - удаляет съеденную еду,
        - увеличивает длину змей, съевших еду,
        - удаляет хвосты у остальных.
 
    Args:
        state (Dict): Игровое состояние (изменяется на месте).
 
    Returns:
        Dict: Изменённое состояние.
    """
    foods = state['board']['food']
    eaten_indices = set()
    for i, snake in enumerate(state['board']['snakes']):
        head = snake['head']
        for food in foods:
            if food['x'] == head['x'] and food['y'] == head['y']:
                eaten_indices.add(i)
                break
 
    # Удаляем съеденную еду
    new_foods = []
    for f in foods:
        if not any(s['head']['x'] == f['x'] and s['head']['y'] == f['y'] for s in state['board']['snakes']):
            new_foods.append(f)
    state['board']['food'] = new_foods
 
    # Обновляем длину и хвосты
    for i, snake in enumerate(state['board']['snakes']):
        if i in eaten_indices:
            snake['length'] += 1
        else:
            if len(snake['body']) > 1:
                snake['body'].pop()
            snake['length'] = len(snake['body'])
    return state
 
def simulate_state(state: Dict, move: str, depth: int) -> Dict:
    """
    Симулирует игру на depth ходов вперёд, начиная с хода нашей змеи = move.
 
    Противники двигаются по умной эвристике (choose_opponent_move_smart).
 
    Args:
        state (Dict): Исходное игровое состояние.
        move (str): Первый ход нашей змеи.
        depth (int): Количество шагов симуляции.
 
    Returns:
        Dict: Новое состояние после depth шагов.
    """
    sim = copy.deepcopy(state)
    for _ in range(depth):
        # Ход нашей змеи
        apply_move_to_snake(sim, sim['you']['id'], move)
        # Ходы противников
        for snake in sim['board']['snakes']:
            if snake['id'] == sim['you']['id']:
                continue
            opp_move = choose_opponent_move_smart(sim, snake)
            apply_move_to_snake(sim, snake['id'], opp_move)
        # Обновление (еда, хвосты)
        sim = update_after_moves(sim)
    return sim
 
def evaluate_state(state: Dict, weights: Dict[str, float]) -> float:
    """
    Оценивает качество состояния по нескольким критериям с заданными весами.
 
    Критерии:
        - доступное пространство (размер компоненты связности),
        - близость к еде,
        - безопасность (риск столкновения),
        - агрессивность (возможность съесть врага),
        - центральность,
        - штраф за малое число свободных соседей (тупик),
        - штраф за близость к углу.
 
    Args:
        state (Dict): Игровое состояние.
        weights (Dict[str, float]): Веса критериев.
 
    Returns:
        float: Общая численная оценка (чем выше, тем лучше).
    """
    my_head = state['you']['head']
    board = state['board']
    health = state['you']['health']
    my_length = state['you']['length']
 
    # 1. Размер связной области (без ограничения по шагам)
    comp_size = component_size(my_head, board)
    max_possible = board['width'] * board['height']
    space_score = comp_size / max_possible if max_possible > 0 else 0
 
    # 2. Еда
    food_dist = min_distance_to_food(my_head, board)
    food_score = 1.0 / (food_dist + 1) if food_dist is not None else 0.0
 
    # 3. Риск столкновения
    risk = compute_risk(state)
    risk_score = 1.0 - risk
 
    # 4. Агрессия
    aggression = compute_aggression(state)
 
    # 5. Центральность
    center_x = board['width'] / 2.0
    center_y = board['height'] / 2.0
    center_dist = abs(my_head['x'] - center_x) + abs(my_head['y'] - center_y)
    center_score = 1.0 - center_dist / (board['width'] + board['height'])
 
    # 6. Штрафы против застревания в углах и тупиках
 
    # 6a. Количество свободных соседей у головы (escape)
    occupied = set()
    for snake in board['snakes']:
        for seg in snake['body']:
            occupied.add((seg['x'], seg['y']))
    escape = 0
    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nx, ny = my_head['x'] + dx, my_head['y'] + dy
        if 0 <= nx < board['width'] and 0 <= ny < board['height']:
            if (nx, ny) not in occupied:
                escape += 1
    # Если escape < 3, это потенциальный тупик – штрафуем (плавно)
    escape_penalty = max(0, (3 - escape) / 3)  # от 0 до 1
 
    # 6b. Близость к углу (чем ближе, тем хуже)
    corners = [(0, 0), (0, board['height']-1), (board['width']-1, 0), (board['width']-1, board['height']-1)]
    min_corner_dist = min(abs(my_head['x'] - cx) + abs(my_head['y'] - cy) for cx, cy in corners)
    # Нормализуем: если дистанция 0..(width+height), то corner_penalty = 1 - min_corner_dist / (width+height)
    corner_penalty = 1.0 - min_corner_dist / (board['width'] + board['height'])
 
    # Корректируем пространственный скоринг с учётом штрафов
    adjusted_space = space_score - 0.3 * corner_penalty - 0.2 * escape_penalty
    adjusted_space = max(0, adjusted_space)  # не может быть отрицательным
 
    score = (weights['space'] * adjusted_space +
             weights['food'] * food_score +
             weights['risk'] * risk_score +
             weights['aggression'] * aggression +
             weights['center'] * center_score)
    return score
 
# ======================================================================
#  НОВАЯ ГЛАВНАЯ ТОЧКА ВХОДА (улучшенный алгоритм)
# ======================================================================
 
def choose_move(game_state: Dict) -> str:
    """
    Главная функция выбора хода.
 
    Реализует улучшенный алгоритм:
        1. Фильтрация безопасных ходов.
        2. Адаптивная глубина симуляции (зависит от доступного пространства).
        3. Симуляция на depth ходов с умным поведением противников.
        4. Оценка каждого симулированного состояния по нескольким критериям
           с динамическими весами и штрафами за тупики.
        5. Выбор хода с максимальной оценкой.
        6. При любой ошибке или отсутствии хода — переход к оригинальной
           логике (модель + эвристика).
 
    Args:
        game_state (Dict): Игровое состояние.
 
    Returns:
        str: Направление хода.
    """
    start_time = time.time()
 
    try:
        # 1. Получаем безопасные ходы
        safe = get_safe_moves(game_state)
        if not safe:
            return choose_move_original(game_state)
        if len(safe) == 1:
            return safe[0]
 
        # 2. Динамические веса
        weights = get_dynamic_weights(game_state)
 
        # 3. Адаптивная глубина симуляции
        board = game_state['board']
        my_head = game_state['you']['head']
        width, height = board['width'], board['height']
        # Оцениваем долю доступного пространства
        comp = component_size(my_head, board)
        space_ratio = comp / (width * height) if width * height > 0 else 0
        num_snakes = len(board['snakes'])
 
        # Если пространства мало – симулируем глубже, чтобы найти выход
        if space_ratio < 0.2 and num_snakes <= 3:
            depth = 3
        elif space_ratio < 0.3 or num_snakes <= 4:
            depth = 2
        else:
            depth = 1
 
        # Ограничиваем глубину, чтобы уложиться по времени (можно увеличить, если позволяет)
        depth = min(depth, 3)
 
        best_move = None
        best_score = -float('inf')
 
        for move in safe:
            sim_state = simulate_state(game_state, move, depth)
            score = evaluate_state(sim_state, weights)
            if score > best_score:
                best_score = score
                best_move = move
 
            # Контроль времени (не более 0.45 сек)
            if time.time() - start_time > 0.45:
                break
 
        if best_move is not None:
            return best_move
    except Exception:
        # При любой ошибке используем оригинальную логику
        pass
 
    # Fallback
    return choose_move_original(game_state)
