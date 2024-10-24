import pygame
import sys
import random
from game_objects import Player, Enemy  # Import the Player and Enemy classes

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
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
BACKGROUND_COLOR = (230, 230, 230)  # Light gray background
UI_COLOR = (50, 50, 50)  # Darker color for UI background
BUTTON_COLOR = (100, 100, 100)  # Button color
BUTTON_HOVER_COLOR = (150, 150, 150)  # Button hover color

# Fonts
font = pygame.font.Font(pygame.font.get_default_font(), 18)

# FPS
clock = pygame.time.Clock()

# Upgrade options
upgrade_options = [
    {"name": "Increase Player Speed", "value": 1},
    {"name": "Increase Bullet Speed", "value": 2},
    {"name": "Increase Player Size (More Health)", "value": 10},
]

# Game variables
player = Player(WIDTH // 2, HEIGHT // 2, 30, RED, 4, 3)
enemies = []
enemy_size = 40
enemy_speed = 2

round_number = 1
enemies_defeated = 0
enemies_to_defeat = round_number * 10

bullet_speed = 7
bullet_cooldown = 300
last_bullet_time = 0
bullet_range = 250

# Enemy spawn variables
spawn_interval = 1000  # Enemy spawn interval in milliseconds (adjust as needed)
last_spawn_time = 0  # Track the last spawn time

# Score tracking
score = 0  # Initialize the score to 0


def spawn_enemy():
    edge = random.choice(['top', 'bottom', 'left', 'right'])  # Randomly select which edge to spawn on
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

    return Enemy(x, y, enemy_size, GREEN, enemy_speed)


# Function to check for collisions between two rectangles
def check_collision(rect1, rect2):
    return pygame.Rect(rect1).colliderect(pygame.Rect(rect2))


def draw_ui():
    """Function to draw the UI elements with improved aesthetics."""
    # Draw UI background
    pygame.draw.rect(screen, UI_COLOR, (10, 10, 100, 80))

    # Health, score, and round text
    health_text = font.render(f"Health: {player.health}", True, WHITE)
    score_text = font.render(f"Score: {score}", True, WHITE)
    round_text = font.render(f"Round: {round_number}", True, WHITE)

    # Draw text on the screen
    screen.blit(health_text, (20, 20))
    screen.blit(score_text, (20, 40))
    screen.blit(round_text, (20, 60))


def draw_upgrade_menu():
    """Function to draw the upgrade menu."""
    menu_width = 300
    menu_height = 250
    menu_x = (WIDTH - menu_width) // 2
    menu_y = (HEIGHT - menu_height) // 2

    # Draw menu background
    pygame.draw.rect(screen, UI_COLOR, (menu_x, menu_y, menu_width, menu_height))

    title_text = font.render("Choose Upgrade", True, WHITE)
    screen.blit(title_text, (menu_x + 20, menu_y + 20))

    # Draw buttons
    for i, option in enumerate(upgrade_options):
        button_rect = pygame.Rect(menu_x + 20, menu_y + 60 + i * 40, 260, 30)
        mouse_pos = pygame.mouse.get_pos()

        # Change button color on hover
        if button_rect.collidepoint(mouse_pos):
            pygame.draw.rect(screen, BUTTON_HOVER_COLOR, button_rect)
        else:
            pygame.draw.rect(screen, BUTTON_COLOR, button_rect)

        # Render button text
        button_text = font.render(option["name"], True, BLACK)
        screen.blit(button_text, (button_rect.x + 5, button_rect.y + 5))

        # Check for button click
        if pygame.mouse.get_pressed()[0] and button_rect.collidepoint(mouse_pos):
            apply_upgrade(option)


def apply_upgrade(option):
    """Apply the selected upgrade to the player."""
    if option["value"] == 1:
        player.speed += 1  # Increase player speed
    elif option["value"] == 2:
        bullet_speed += 2  # Increase bullet speed
    elif option["value"] == 10:
        player.size += 5  # Increase player size (health)

    # Reset round variables for the next round
    global round_number, enemies_defeated, enemies_to_defeat
    round_number += 1
    enemies_defeated = 0
    enemies_to_defeat = round_number * 10


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


def update_game_objects():
    """Function to update all game objects."""
    # Player movement
    keys = pygame.key.get_pressed()
    player.move(keys, WIDTH, HEIGHT)

    # Bullet shooting with cooldown
    current_time = pygame.time.get_ticks()
    global last_bullet_time
    last_bullet_time = player.shoot(keys, bullet_speed, bullet_cooldown, current_time, last_bullet_time)

    # Move bullets
    for bullet in player.bullets[:]:
        bullet[0] += bullet[2]  # Move horizontally
        bullet[1] += bullet[3]  # Move vertically
        distance_traveled = ((bullet[0] - bullet[4]) ** 2 + (bullet[1] - bullet[5]) ** 2) ** 0.5

        # Remove bullet if it travels beyond its range or goes off-screen
        if distance_traveled > bullet_range or bullet[0] < 0 or bullet[0] > WIDTH or bullet[1] < 0 or bullet[
            1] > HEIGHT:
            player.bullets.remove(bullet)

    # Spawn enemies based on the spawn interval
    global last_spawn_time
    if current_time - last_spawn_time > spawn_interval:
        enemies.append(spawn_enemy())
        last_spawn_time = current_time  # Reset spawn timer

    # Move enemies towards player
    for enemy in enemies:
        enemy.move_towards_player(player.x, player.y)


def check_collisions():
    """Function to check for collisions between bullets and enemies, and between the player and enemies."""
    global enemies_defeated, score
    # Check bullet-enemy collisions
    for bullet in player.bullets[:]:
        bullet_rect = (bullet[0], bullet[1], 10, 10)
        for enemy in enemies[:]:
            enemy_rect = (enemy.x, enemy.y, enemy.size, enemy.size)
            if check_collision(bullet_rect, enemy_rect):
                player.bullets.remove(bullet)  # Remove bullet
                enemies.remove(enemy)  # Remove enemy
                enemies_defeated += 1
                score += 1  # Increase score by 1 for every kill

    # Check player-enemy collisions
    player_rect = (player.x, player.y, player.size, player.size)
    for enemy in enemies[:]:
        enemy_rect = (enemy.x, enemy.y, enemy.size, enemy.size)
        if check_collision(player_rect, enemy_rect):
            player.health -= 1  # Decrease player health
            enemies.remove(enemy)  # Remove enemy that hit the player
            if player.health <= 0:
                return False  # End game if player health is 0
    return True


# Game loop
running = True
while running:
    screen.fill(BACKGROUND_COLOR)
    handle_input()

    # Check if all enemies are defeated to show upgrade menu
    if enemies_defeated >= enemies_to_defeat:
        draw_upgrade_menu()
    else:
        update_game_objects()
        running = check_collisions()

        # Draw game objects
        player.draw(screen)
        for bullet in player.bullets:
            pygame.draw.rect(screen, BLACK, (bullet[0], bullet[1], 10, 10))
        for enemy in enemies:
            enemy.draw(screen)

    draw_ui()  # Draw UI at the end of each frame

    pygame.display.flip()
    clock.tick(60)

pygame.quit()