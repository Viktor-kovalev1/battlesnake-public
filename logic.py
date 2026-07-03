 
import random
from collections import deque
from typing import Dict, List, Set, Tuple
 
def info() -> Dict[str, str]:
    return {
        "apiversion": "1",
        "author": "Alex2034",
        "color": "#1E1E24",  # Агрессивный графитовый
        "head": "shades",
        "tail": "bolt",
    }
 
def start(game_state: Dict):
    print("GAME START")
 
def end(game_state: Dict):
    print("GAME OVER\n")
 
def get_bfs_space(start_pos: Tuple[int, int], obstacles: Set[Tuple[int, int]], width: int, height: int, max_depth: int) -> int:
    """Продвинутый Flood Fill: оценивает реальную емкость зоны"""
    if start_pos in obstacles:
        return 0
        
    queue = deque([start_pos])
    visited = {start_pos}
    space_count = 0
 
    while queue and space_count < max_depth:
        curr = queue.popleft()
        space_count += 1
 
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = curr[0] + dx, curr[1] + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbor = (nx, ny)
                if neighbor not in obstacles and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
    return space_count
 
def move(game_state: Dict) -> Dict[str, str]:
    board = game_state["board"]
    my_snake = game_state["you"]
    my_id = my_snake["id"]
    my_head = (my_snake["head"]["x"], my_snake["head"]["y"])
    my_length = my_snake["length"]
    
    width = board["width"]
    height = board["height"]
    foods = [(f["x"], f["y"]) for f in board["food"]]
 
    # 1. Сбор динамических препятствий с учетом механики хвостов
    obstacles: Set[Tuple[int, int]] = set()
    
    for snake in board["snakes"]:
        body = snake["body"]
        # Если змея только появилась или в ней 1 клетка (технически маловероятно, но для безопасности)
        if len(body) < 2:
            for part in body:
                obstacles.add((part["x"], part["y"]))
            continue
 
        # Логика хвоста: если на прошлом ходу змея съела еду (здоровье восстановилось до 100),
        # её хвост НЕ сдвинется на этом ходу. Если не ела — хвост сдвинется и клетка станет свободной.
        # В Battlesnake здоровье падает на 1 каждый ход, а при съедании еды становится ровно 100.
        # state["turn"] == 0 — старт игры, там проверки здоровья специфичны.
        will_grow = (snake["health"] == 100 and game_state["turn"] > 0)
        
        # Заносим все тело, кроме хвоста (если змея не растет)
        parts_to_add = body if will_grow else body[:-1]
        for part in parts_to_add:
            obstacles.add((part["x"], part["y"]))
 
    # 2. Зоны смертельного риска и зоны доминирования (Head-to-Head)
    dangerous_zones: Set[Tuple[int, int]] = set()
    kill_zones: Set[Tuple[int, int]] = set()
 
    for snake in board["snakes"]:
        if snake["id"] == my_id:
            continue
            
        enemy_head = (snake["head"]["x"], snake["head"]["y"])
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = enemy_head[0] + dx, enemy_head[1] + dy
            if 0 <= nx < width and 0 <= ny < height:
                if snake["length"] >= my_length:
                    dangerous_zones.add((nx, ny))  # Враг равен или длиннее — уходим
                else:
                    kill_zones.add((nx, ny))       # Враг меньше — можем задушить!
 
    # 3. Анализ векторов движения
    directions = {
        "up": (my_head[0], my_head[1] + 1),
        "down": (my_head[0], my_head[1] - 1),
        "left": (my_head[0] - 1, my_head[1]),
        "right": (my_head[0] + 1, my_head[1])
    }
 
    best_move = "down"
    best_score = -9999999
 
    for direction, target_pos in directions.items():
        tx, ty = target_pos
 
        # Исключаем мгновенный суицид об стены
        if tx < 0 or tx >= width or ty < 0 or ty >= height:
            continue
 
        # Исключаем мгновенный суицид об тела (с учетом сдвига хвостов!)
        if target_pos in obstacles:
            continue
 
        # Симуляция: добавляем шаг во временные препятствия
        temp_obstacles = obstacles.copy()
        temp_obstacles.add(target_pos)
        
        # Считаем емкость пространства. Глубина поиска равна длине нашего тела + запас
        available_space = get_bfs_space(target_pos, temp_obstacles, width, height, max_depth=my_length + 10)
 
        # Высчитываем жесткий штраф за замкнутые пространства
        if available_space < my_length:
            # Смертельная ловушка, если пространства критически мало
            space_score = (my_length - available_space) * -500
        else:
            space_score = available_space * 15
 
        # Базовый вес хода
        move_score = space_score
 
        # Оценка лобовых столкновений
        if target_pos in dangerous_zones:
            move_score -= 3000  # Колоссальный штраф, змея выберет этот ход только ради выживания
        if target_pos in kill_zones:
            move_score += 150   # Хороший стимул сожрать бедолагу
 
        # Умный скоринг еды
        if foods:
            # Манхэттенское расстояние до ближайшей еды из целевой точки
            min_food_dist = min(abs(tx - fx) + abs(ty - fy) for fx, fy in foods)
            
            # Проверяем, не ближе ли враги к этой еде, чем мы
            enemy_closer_to_food = False
            for snake in board["snakes"]:
                if snake["id"] == my_id:
                    continue
                ex, ey = snake["head"]["x"], snake["head"]["y"]
                enemy_dist = min(abs(ex - fx) + abs(ey - fy) for fx, fy in foods)
                if enemy_dist < min_food_dist:
                    enemy_closer_to_food = True
                    break
 
            # Корректируем ценность еды в зависимости от голода
            if my_snake["health"] < 35:
                # Включаем режим выживания — еда любой ценой
                move_score += (100 - min_food_dist) * 15
            else:
                # Если сыты, но еда свободна (враги далеко) — забираем её для доминирования в длине
                if not enemy_closer_to_food:
                    move_score += (100 - min_food_dist) * 2
                else:
                    move_score += (100 - min_food_dist) * 0.2
 
        # Дополнительный микро-бонус за удержание центра карты в начале игры (чтобы не зажиматься у стен)
        center_x, center_y = width // 2, height // 2
        dist_to_center = abs(tx - center_x) + abs(ty - center_y)
        move_score += (100 - dist_to_center) * 0.1
 
        # Обновление лучшего хода
        if move_score > best_score:
            best_score = move_score
            best_move = direction
 
    # Защитный механизм: если все ходы ведут к смерти, выбираем любой не-суицидальный случайный
    if best_score < -500000:
        valid_moves = []
        for direction, target_pos in directions.items():
            tx, ty = target_pos
            if 0 <= tx < width and 0 <= ty < height and target_pos not in obstacles:
                valid_moves.append(direction)
        if valid_moves:
            best_move = random.choice(valid_moves)
 
    return {"move": best_move}
