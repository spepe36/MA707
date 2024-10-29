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
    return pygame.Rect(rect1).colliderect(pygame.Rect(rect2))


def check_collisions():
    """Function to check for collisions between bullets and enemies, and between the player and enemies."""
    global enemies_defeated, score
    # Check bullet-enemy collisions
    for single_bullet in player.bullets[:]:
        bullet_rect = (single_bullet[0], single_bullet[1], 10, 10)
        for single_enemy in enemies[:]:
            enemy_rect = (single_enemy.x, single_enemy.y, single_enemy.size, single_enemy.size)
            if check_collision(bullet_rect, enemy_rect):
                player.bullets.remove(single_bullet)  # Remove bullet
                enemies.remove(single_enemy)  # Remove enemy
                enemies_defeated += 1
                score += 1  # Increase score by 1 for every kill

    # Check player-enemy collisions
    player_rect = (player.x, player.y, player.size, player.size)
    for single_enemy in enemies[:]:
        enemy_rect = (single_enemy.x, single_enemy.y, single_enemy.size, single_enemy.size)
        if check_collision(player_rect, enemy_rect):
            player.health -= 1  # Decrease player health
            enemies.remove(single_enemy)  # Remove enemy that hit the player
            if player.health <= 0:
                return False  # End game if player health is 0
    return True


def enemy_selected():
    """Function to choose the type of enemy to spawn based on the round's spawn rates."""
    spawn_rates = {1: {'yellow': 1},
                   2: {'yellow': 7, 'green': 3},
                   3: {'yellow': 5, 'green': 3, 'red': 2},
                   4: {'yellow': 2, 'green': 5, 'red': 3},
                   5: {'yellow': 1, 'green': 2, 'red': 3, 'purple': 4}}

    if round_number > 5:
        current_rates = spawn_rates[5]
    else:
        current_rates = spawn_rates[round_number]

    color_array = []

    # Populate color_array according to spawn rates
    for color, spawn_rate in current_rates.items():
        color_array.extend([color] * spawn_rate)

    return random.choice(color_array)


def spawn_enemy():
    """Function to spawn enemies at a given location."""

    if len(enemies) >= max_enemies:
        return None

    edge = random.choice(['top', 'bottom', 'left', 'right'])

    enemy_color = enemy_selected()

    enemy_information = {
        'yellow': [30, 2.6, (0, 255, 0)],
        'green': [15, 3.0, (0, 0, 255)],
        'red': [60, 2, (255, 0, 0)],
        'purple': [40, 3.3, (255, 0, 255)],
    }

    enemy_size = enemy_information[enemy_color][0]
    enemy_speed = enemy_information[enemy_color][1]
    enemy_color = enemy_information[enemy_color][2]

    x, y = 0, 0

    if edge == 'top':
        x = random.randint(0, WIDTH - enemy_size)
        y = 0  # Spawn at the top edge
    elif edge == 'bottom':
        x = random.randint(0, WIDTH - enemy_size)
        y = HEIGHT - enemy_size  # Spawn at the bottom edge
    elif edge == 'left':
        x = 0  # Spawn at the left edge
        y = random.randint(0, HEIGHT - enemy_size)
    elif edge == 'right':
        x = WIDTH - enemy_size  # Spawn at the right edge
        y = random.randint(0, HEIGHT - enemy_size)

    return Enemy(x, y, enemy_size, enemy_color, enemy_speed)


def draw_ui():
    """Function to draw the UI elements with improved aesthetics."""
    pygame.draw.rect(screen, UI_COLOR, (10, 10, 100, 60))

    # Score, and round text
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

    # Draw buttons for selected upgrades
    for i, option in enumerate(upgrades):
        button_rect = pygame.Rect(menu_x + 20, menu_y + 60 + i * 40, 260, 30)
        mouse_pos = pygame.mouse.get_pos()

        # Change button color on hover
        if button_rect.collidepoint(mouse_pos):
            pygame.draw.rect(screen, BUTTON_HOVER_COLOR, button_rect)
        else:
            pygame.draw.rect(screen, BUTTON_COLOR, button_rect)

        # Render button text
        upgrade_name = list(option.keys())[0]  # Get the upgrade name
        button_text = font.render(upgrade_name, True, BLACK)
        screen.blit(button_text, (button_rect.x + 5, button_rect.y + 5))

        # Check for button click
        if pygame.mouse.get_pressed()[0] and button_rect.collidepoint(mouse_pos):
            apply_upgrade(option)

def select_random_upgrades():
    """Function to select three random upgrades from the upgrade options."""
    three_upgrades = random.sample(upgrade_options, 3)
    return three_upgrades


def apply_upgrade(option):
    """Apply the selected upgrade to the player."""
    global round_number, enemies_defeated, enemies_to_defeat, spawn_interval, bullet_speed, bullet_range, bullet_cooldown, upgrade_menu_open

    # Extract the upgrade name and value from the option
    upgrade_name = list(option.keys())[0]  # Get the upgrade name
    upgrade_value = option[upgrade_name]   # Get the corresponding value

    if upgrade_name == "Character Speed":
        player.speed += upgrade_value  # Increase player speed
    elif upgrade_name == "Bullet Speed":
        bullet_speed += upgrade_value  # Increase bullet speed
    elif upgrade_name == "Character Size":
        player.size += upgrade_value  # Increase player size (health)
    elif upgrade_name == "Bullet Size":
        player.bullet_size += upgrade_value  # Increase bullet size
    elif upgrade_name == "Bullet Range":
        bullet_range += upgrade_value  # Increase bullet range
    elif upgrade_name == "Attack Speed":
        bullet_cooldown += upgrade_value

    # After applying the upgrade, update the round state
    round_number += 1
    enemies_defeated = 0
    enemies_to_defeat = round_number * 10
    spawn_interval -= 100

    # Reset player position to the center of the screen
    player.x, player.y = WIDTH // 2, HEIGHT // 2

    # Clear all existing enemies
    enemies.clear()
    player.bullets.clear()

    # Close the upgrade menu
    upgrade_menu_open = False

def update_game_objects():
    """Function to update all game objects."""
    # Player movement
    keys = pygame.key.get_pressed()
    player.move(keys, WIDTH, HEIGHT)

    # Bullet shooting with cooldown
    current_time = pygame.time.get_ticks()
    global last_bullet_time

    # Player.avoid_enemies(player, enemies)

    last_bullet_time = player.shoot(keys, bullet_speed, bullet_cooldown, current_time, last_bullet_time)

    # Move bullets
    for single_bullet in player.bullets[:]:
        single_bullet[0] += single_bullet[2]  # Move horizontally
        single_bullet[1] += single_bullet[3]  # Move vertically
        distance_traveled = ((single_bullet[0] - single_bullet[4]) ** 2 + (single_bullet[1] - single_bullet[5]) ** 2) ** 0.5

        # Remove bullet if it travels beyond its range or goes off-screen
        if distance_traveled > bullet_range or single_bullet[0] < 0 or single_bullet[0] > WIDTH or single_bullet[1] < 0 or single_bullet[
            1] > HEIGHT:
            player.bullets.remove(single_bullet)

    # Spawn enemies based on the spawn interval
    global last_spawn_time
    if current_time - last_spawn_time > spawn_interval:
        new_enemy = spawn_enemy()
        if new_enemy is not None:
            enemies.append(new_enemy)
        last_spawn_time = current_time

    # Move enemies towards player
    for single_enemy in enemies:
        single_enemy.move_towards_player(player.x, player.y, enemies)


# Initialize Pygame
pygame.init()

# Get the display resolution
info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h

# Enable windowed fullscreen (borderless)
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.NOFRAME)

pygame.display.set_caption("Square Shooter")

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
player = Player(WIDTH // 2, HEIGHT // 2, 30, RED, 4, 1, WIDTH, HEIGHT, 5)
enemies = []

# Round Variables
round_number = 1
enemies_defeated = 0
enemies_to_defeat = round_number * 10
max_enemies = 35

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

# Game loop
running = True
while running:
    screen.fill(BACKGROUND_COLOR)
    handle_input()

    if enemies_defeated >= enemies_to_defeat and not upgrade_menu_open:
        selected_upgrades = select_random_upgrades()
        upgrade_menu_open = True  # Open the upgrade menu

    if upgrade_menu_open:
        draw_upgrade_menu(selected_upgrades)
    else:
        update_game_objects()
        running = check_collisions()
        player.draw(screen)

        for bullet in player.bullets:
            pygame.draw.rect(screen, BLACK, (bullet[0], bullet[1], 10, 10))

        for enemy in enemies:
            enemy.draw(screen)

    draw_ui()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()