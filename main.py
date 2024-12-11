import pygame
import sys
import random
from game_objects import Player, Enemy


def handle_input():
    """Function to handle user input."""

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()


def check_collision(rect1, rect2):
    """Function to check if two rectangles are colliding."""

    return pygame.Rect(rect1).colliderect(pygame.Rect(rect2))


last_score_update = 0


def check_collisions():
    """Function to check for collisions between bullets and enemies, and between the player and enemies."""

    global enemies_defeated, score

    # Bullet-Enemy collisions
    for single_bullet in player.bullets[:]:
        bullet_rect = (single_bullet[0], single_bullet[1], 10, 10)
        for single_enemy in enemies[:]:
            enemy_rect = (single_enemy.x, single_enemy.y, single_enemy.size, single_enemy.size)
            if check_collision(bullet_rect, enemy_rect):
                player.bullets.remove(single_bullet)
                enemies.remove(single_enemy)
                enemies_defeated += 1
                # Removed the line that increments the score when an enemy is defeated by a bullet
                # score += 1  # <--- REMOVE THIS LINE

    # Player-Enemy collisions
    player_rect = (player.x, player.y, player.size, player.size)
    for single_enemy in enemies[:]:
        enemy_rect = (single_enemy.x, single_enemy.y, single_enemy.size, single_enemy.size)
        if check_collision(player_rect, enemy_rect):
            player.health -= 1
            enemies.remove(single_enemy)
            if player.health <= 0:
                return False
    return True


def enemy_selected():
    """Function to choose the type of enemy to spawn based on the round's spawn rates."""

    spawn_rates = {1: {'yellow': 1},  # 100% of enemies will be yellow
                   2: {'yellow': 1},  # 100% of enemies will be yellow, 30% will be green
                   3: {'yellow': 5, 'green': 3, 'red': 2},
                   4: {'yellow': 2, 'green': 5, 'red': 3},
                   5: {'yellow': 1, 'green': 2, 'red': 3, 'purple': 4}}

    if round_number > 5:
        current_rates = spawn_rates[5]
    else:
        current_rates = spawn_rates[round_number]

    color_array = []
    for color, spawn_rate in current_rates.items():
        color_array.extend([color] * spawn_rate)

    return random.choice(color_array)


def spawn_enemy():
    """Function to spawn enemies at a given location."""

    enemy_information = {
        'yellow': [30, 2.6, (0, 255, 0)],  # [Size, speed, color]
        'green': [15, 3.0, (0, 0, 255)],
        'red': [60, 2, (255, 0, 0)],
        'purple': [40, 3.3, (255, 0, 255)],
    }

    edge = random.choice(['top', 'bottom', 'left', 'right'])

    if len(enemies) >= max_enemies:
        return None

    enemy_color = enemy_selected()
    enemy_image = 'enemy_image.png'
    enemy_size = enemy_information[enemy_color][0]
    enemy_speed = enemy_information[enemy_color][1]
    enemy_color = enemy_information[enemy_color][2]

    x, y = 0, 0

    # Spawn enemy at selected edge
    if edge == 'top':
        x = random.randint(0, WIDTH - enemy_size)
        y = 0
    elif edge == 'bottom':
        x = random.randint(0, WIDTH - enemy_size)
        y = HEIGHT - enemy_size
    elif edge == 'left':
        x = 0
        y = random.randint(0, HEIGHT - enemy_size)
    elif edge == 'right':
        x = WIDTH - enemy_size
        y = random.randint(0, HEIGHT - enemy_size)

    return Enemy(x, y, enemy_size, enemy_color, enemy_speed, enemy_image)


def draw_ui():
    """Function to draw the UI elements with improved aesthetics."""

    pygame.draw.rect(screen, UI_COLOR, (10, 10, 100, 60))

    score_text = font.render(f"Score: {score}", True, WHITE)
    round_text = font.render(f"Round: {round_number}", True, WHITE)
    screen.blit(score_text, (20, 20))
    screen.blit(round_text, (20, 40))


def draw_upgrade_menu(upgrades):
    """Function to draw the upgrade menu."""

    menu_width = 300
    menu_height = 250
    menu_x = (WIDTH - menu_width) // 2
    menu_y = (HEIGHT - menu_height) // 2

    pygame.draw.rect(screen, UI_COLOR, (menu_x, menu_y, menu_width, menu_height))

    title_text = font.render("Choose Upgrade", True, WHITE)
    screen.blit(title_text, (menu_x + 20, menu_y + 20))

    for i, option in enumerate(upgrades):
        button_rect = pygame.Rect(menu_x + 20, menu_y + 60 + i * 40, 260, 30)
        mouse_pos = pygame.mouse.get_pos()

        if button_rect.collidepoint(mouse_pos):
            pygame.draw.rect(screen, BUTTON_HOVER_COLOR, button_rect)
        else:
            pygame.draw.rect(screen, BUTTON_COLOR, button_rect)

        upgrade_name = list(option.keys())[0]
        button_text = font.render(upgrade_name, True, BLACK)
        screen.blit(button_text, (button_rect.x + 5, button_rect.y + 5))

        if pygame.mouse.get_pressed()[0] and button_rect.collidepoint(mouse_pos):
            apply_upgrade(option)


def select_random_upgrades():
    """Function to select three random upgrades from the upgrade options."""

    return random.sample(upgrade_options, 3)


def apply_upgrade(option):
    """Apply the selected upgrade to the player."""

    global round_number, enemies_defeated, enemies_to_defeat, spawn_interval, bullet_speed, bullet_range, bullet_cooldown, upgrade_menu_open

    upgrade_name = list(option.keys())[0]
    upgrade_value = option[upgrade_name]

    match upgrade_name:
        case "Character Speed":
            player.speed += upgrade_value
        case "Bullet Speed":
            bullet_speed += upgrade_value
        case "Character Size":
            player.size += upgrade_value
        case "Bullet Size":
            player.bullet_size += upgrade_value
        case "Bullet Range":
            bullet_range += upgrade_value
        case "Attack Speed":
            bullet_cooldown += upgrade_value

    round_number += 1
    enemies_defeated = 0
    enemies_to_defeat = round_number * 10
    spawn_interval -= 100

    # Reset player position to the center of the screen
    player.x, player.y = WIDTH // 2, HEIGHT // 2

    enemies.clear()
    player.bullets.clear()

    upgrade_menu_open = False


def update_game_objects():
    """Function to update all game objects."""

    global last_bullet_time, last_spawn_time, last_score_update, score, round_number, enemies_defeated, enemies_to_defeat, spawn_interval, max_enemies

    keys = pygame.key.get_pressed()
    player.move(keys, WIDTH, HEIGHT)

    current_time = pygame.time.get_ticks()

    # Update bullet positions
    last_bullet_time = player.shoot(keys, bullet_speed, bullet_cooldown, current_time, last_bullet_time)

    for single_bullet in player.bullets[:]:
        single_bullet[0] += single_bullet[2]
        single_bullet[1] += single_bullet[3]
        distance_traveled = ((single_bullet[0] - single_bullet[4]) ** 2 + (single_bullet[1] - single_bullet[5]) ** 2) ** 0.5

        if (distance_traveled > bullet_range or
            single_bullet[0] < 0 or single_bullet[0] > WIDTH or
            single_bullet[1] < 0 or single_bullet[1] > HEIGHT):
            player.bullets.remove(single_bullet)

    # Spawn enemies
    if current_time - last_spawn_time > spawn_interval:
        new_enemy = spawn_enemy()
        if new_enemy is not None:
            enemies.append(new_enemy)
        last_spawn_time = current_time

    # Move enemies
    for single_enemy in enemies:
        single_enemy.move_towards_player(player.x, player.y, enemies)

    # Increment score every second
    if current_time - last_score_update >= 1000:  # 1000 ms = 1 second
        score += 1
        last_score_update = current_time

    # Check if we should advance to round 2
    if score >= 30 and round_number == 1:
        # Advance to round 2
        round_number = 2
        enemies_defeated = 0
        enemies_to_defeat = round_number * 10
        spawn_interval -= 100
        max_enemies+=10

        # Optional: Reset player position
        player.x, player.y = WIDTH // 2, HEIGHT // 2

        # Clear existing enemies and bullets to start fresh
        enemies.clear()
        player.bullets.clear()


def show_start_screen(screen1, title_font1, button_font1, player1, enemies1):
    """Display the start screen with the game as the background."""

    while True:
        # Render the game as the background
        screen1.fill((230, 230, 230))

        # Draw the player
        player1.draw(screen1)

        # Draw a couple of enemies
        for selected_enemy in enemies1:
            selected_enemy.draw(screen1)

        # Draw the title
        title_text = title_font1.render("Purrsuit", True, (255, 100, 100))
        title_rect = title_text.get_rect(center=(WIDTH // 2, HEIGHT // 4))
        screen1.blit(title_text, title_rect)

        # Draw the Start Game button
        button_width, button_height = 200, 60
        button_x = (WIDTH - button_width) // 2
        button_y = HEIGHT // 1.5
        button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

        pygame.draw.rect(screen1, (100, 100, 200), button_rect)
        button_text = button_font1.render("Start Game", True, (255, 255, 255))
        button_text_rect = button_text.get_rect(center=button_rect.center)
        screen1.blit(button_text, button_text_rect)

        # Display the screen
        pygame.display.flip()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    return  #


def show_game_over_screen(screen1, title_font1, button_font1):
    """Display the game over screen without changing the game state."""

    while True:
        # Draw Game Over Title
        title_text = title_font1.render("Game Over", True, (255, 0, 0))  # Red title
        title_rect = title_text.get_rect(center=(WIDTH // 2, HEIGHT // 3))
        screen1.blit(title_text, title_rect)

        # Draw Restart Button
        button_width, button_height = 200, 60
        button_x = (WIDTH - button_width) // 2
        button_y = HEIGHT // 2
        restart_button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

        pygame.draw.rect(screen1, (0, 200, 0), restart_button_rect)
        restart_text = button_font1.render("Restart", True, (255, 255, 255))
        restart_text_rect = restart_text.get_rect(center=restart_button_rect.center)
        screen1.blit(restart_text, restart_text_rect)

        # Draw Quit Button
        quit_button_rect = pygame.Rect(button_x, button_y + 80, button_width, button_height)

        pygame.draw.rect(screen1, (200, 0, 0), quit_button_rect)
        quit_text = button_font1.render("Quit", True, (255, 255, 255))
        quit_text_rect = quit_text.get_rect(center=quit_button_rect.center)
        screen1.blit(quit_text, quit_text_rect)

        # Update display
        pygame.display.flip()

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if restart_button_rect.collidepoint(event.pos):
                    return "restart"
                elif quit_button_rect.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()


# Initialize Pygame
pygame.init()
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)

# Pygame Name
pygame.display.set_caption("Purrsuit")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BACKGROUND_COLOR = (230, 230, 230)
UI_COLOR = (50, 50, 50)
BUTTON_COLOR = (100, 100, 100)
BUTTON_HOVER_COLOR = (150, 150, 150)

# Fonts
font = pygame.font.Font(pygame.font.get_default_font(), 18)

# FPS
clock = pygame.time.Clock()

# Upgrade options
upgrade_options = [
    {"Character Speed": 0.3},
    {"Bullet Speed": -2},
    {"Character Size": -10},
    {"Bullet Size": 2},
    {"Bullet Range": 10},
    {"Attack Speed": -20}
]

# Game variables
player = Player(WIDTH // 2, HEIGHT // 2, 30, RED, 6, 1, WIDTH, HEIGHT, 5, image_path='player_image.png')
enemies = []

# Round Variables
round_number = 1
enemies_defeated = 0
enemies_to_defeat = round_number * 10
max_enemies = 20

# Bullet variables
bullet_speed = 7
bullet_cooldown = 300
last_bullet_time = 0
bullet_range = 250
bullet_size = 2

# Enemy spawn variables
spawn_interval = 600
last_spawn_time = 0

# Score tracking
score = 0

upgrade_menu_open = False
selected_upgrades = []

# Enemies placed for starting screen
start_screen_enemies = [
    Enemy(WIDTH // 3, HEIGHT // 3, 30, (0, 255, 0), 2, image_path='enemy_image.png'),
    Enemy(2 * WIDTH // 3, 2 * HEIGHT // 3, 30, (0, 0, 255), 2, image_path='enemy_image.png'),
    Enemy(WIDTH // 2.8, HEIGHT // 1.2, 30, (0, 0, 255), 2, image_path='enemy_image.png'),
    Enemy(WIDTH // 1.7, HEIGHT // 2.8, 30, (0, 0, 255), 2, image_path='enemy_image.png')
]

title_font = pygame.font.Font(pygame.font.get_default_font(), 72)
button_font = pygame.font.Font(pygame.font.get_default_font(), 36)

# Show the start screen
show_start_screen(screen, title_font, button_font, player, start_screen_enemies)

# Game loop
running = True

while running:
    screen.fill(BACKGROUND_COLOR)
    handle_input()

    # Update game objects and check collisions
    update_game_objects()
    if not check_collisions():
        # Draw all game objects in their current state
        player.draw(screen)
        for enemy in enemies:
            enemy.draw(screen)

        draw_ui()

        # Show game over screen
        game_over_action = show_game_over_screen(screen, title_font, button_font)
        if game_over_action == "restart":
            player.health = 1
            player.x, player.y = WIDTH // 2, HEIGHT // 2
            enemies.clear()
            enemies_defeated = 0
            round_number = 1
            score = 0
            spawn_interval = 600
        else:
            running = False

    else:
        player.draw(screen)

        for bullet in player.bullets:
            pygame.draw.rect(screen, BLACK, (bullet[0], bullet[1], 10, 10))

        for enemy in enemies:
            enemy.draw(screen)

        draw_ui()

    pygame.display.flip()
    clock.tick(60)
