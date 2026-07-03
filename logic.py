import random
from collections import deque
from typing import Dict, List, Set, Tuple
 
def info() -> Dict[str, str]:
    print("INFO")
    return {
        "apiversion": "1",
        "author": "Alex2034",
        "color": "#8855ff",  # Стильный фиолетовый
        "head": "shades",    # Змея в очках
        "tail": "bolt",      # Хвост-молния
    }
 
def start(game_state: Dict):
    print("GAME START")
 
def end(game_state: Dict):
    print("GAME OVER\n")
 
def get_bfs_space(start_pos: Tuple[int, int], obstacles: Set[Tuple[int, int]], width: int, height: int, max_depth: int = 40) -> int:
    """Считает количество доступных свободных клеток методом BFS (Flood Fill)"""
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
    my_head = (my_snake["head"]["x"], my_snake["head"]["y"])
    my_length = my_snake["length"]
    
    width = board["width"]
    height = board["height"]
 
    # 1. Собираем все базовые препятствия (стены и тела змей) в set для мгновенного поиска
    obstacles: Set[Tuple[int, int]] = set()
    for snake in board["snakes"]:
        # Хвост змеи сдвинется на следующем ходу, если змея не съела еду.
        # Для простоты сейчас заносим все тело.
        for part in snake["body"]:
            obstacles.add((part["x"], part["y"]))
 
    # 2. Анализируем головы врагов для предсказания лобовых столкновений
    dangerous_zones: Set[Tuple[int, int]] = set()
    kill_zones: Set[Tuple[int, int]] = set()
 
    for snake in board["snakes"]:
        if snake["id"] == my_snake["id"]:
            continue
            
        enemy_head = (snake["head"]["x"], snake["head"]["y"])
        # Считаем клетки, куда враг может пойти на следующем ходу
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = enemy_head[0] + dx, enemy_head[1] + dy
            if 0 <= nx < width and 0 <= ny < height:
                if snake["length"] >= my_length:
                    dangerous_zones.add((nx, ny)) # Враг длиннее или равен нам — это смерть
                else:
                    kill_zones.add((nx, ny))      # Враг меньше — мы можем его съесть!
 
    # 3. Оцениваем каждый из 4 возможных ходов
    directions = {
        "up": (my_head[0], my_head[1] + 1),
        "down": (my_head[0], my_head[1] - 1),
        "left": (my_head[0] - 1, my_head[1]),
        "right": (my_head[0] + 1, my_head[1])
    }
 
    best_move = "down"
    best_score = -999999
 
    foods = [(f["x"], f["y"]) for f in board["food"]]
 
    for direction, target_pos in directions.items():
        tx, ty = target_pos
 
        # Шаг 3.1: Проверка железных границ карты
        if tx < 0 or tx >= width or ty < 0 or ty >= height:
            continue
 
        # Шаг 3.2: Проверка на мгновенное самоубийство о тело змеи
        if target_pos in obstacles:
            continue
 
        # Шаг 3.3: Считаем доступное пространство (Flood Fill)
        # Временно добавляем эту клетку в препятствия для честного расчета маневра из нее
        temp_obstacles = obstacles.copy()
        temp_obstacles.add(target_pos)
        available_space = get_bfs_space(target_pos, temp_obstacles, width, height, max_depth=my_length + 5)
 
        # Если места меньше, чем длина нашего тела — это потенциальная ловушка
        if available_space < my_length:
            space_penalty = (my_length - available_space) * -100
        else:
            space_penalty = 0
 
        # Шаг 3.4: Рассчитываем базовые очки для этого хода
        move_score = available_space * 10 + space_penalty
 
        # Шаг 3.5: Учитываем опасные зоны голов соперников
        if target_pos in dangerous_zones:
            move_score -= 500  # Жесткий штраф за риск проиграть дуэль
        if target_pos in kill_zones:
            move_score += 50   # Бонус за возможность убить мелкую змею
 
        # Шаг 3.6: Охота за едой (если голодны или еда совсем близко)
        if foods:
            # Находим расстояние до ближайшей еды
            min_food_dist = min(abs(tx - fx) + abs(ty - fy) for fx, fy in foods)
            
            if my_snake["health"] < 40:
                # Критически голодны — приоритет еде максимальный
                move_score += (100 - min_food_dist) * 5
            else:
                # Сыты — просто мягко подталкиваем в сторону еды, если она по пути
                move_score += (100 - min_food_dist) * 0.5
 
        # Выбираем ход с наибольшим количеством очков
        if move_score > best_score:
            best_score = move_score
            best_move = direction
 
    print(f"MOVE {game_state['turn']}: {best_move} (Score: {best_score})")
    return {"move": best_move}
