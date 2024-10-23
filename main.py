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
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# FPS
clock = pygame.time.Clock()

# Upgrade options
upgrade_options = [
    {"name": "Increase Player Speed", "value": 1},
    {"name": "Increase Bullet Speed", "value": 2},
    {"name": "Increase Player Size (More Health)", "value": 10},
]

# Function to check for collisions between two rectangles
def check_collision(rect1, rect2):
    return pygame.Rect(rect1).colliderect(pygame.Rect(rect2))

# Function to handle round progression
def next_round():
    global round_number, enemies_to_defeat, enemies_defeated, enemy_speed
    round_number += 1
    enemies_defeated = 0
    enemies_to_defeat = round_number * 10
    enemy_speed += 0.5  # Increase enemy speed each round

# Function to offer upgrade choices
def offer_upgrades():
    global player_speed, bullet_speed, player_size, player_health
    selected_upgrade = None

    # Font for the button text
    font = pygame.font.SysFont(None, 36)

    while selected_upgrade is None:
        screen.fill(WHITE)

        # Event handling for mouse input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left mouse button
                mouse_x, mouse_y = event.pos

                # Check if the mouse is over any of the upgrade buttons
                for i, upgrade in enumerate(upgrade_options):
                    button_rect = pygame.Rect(100, 100 + i * 50, 600, 40)
                    if button_rect.collidepoint((mouse_x, mouse_y)):
                        selected_upgrade = upgrade["value"]
                        break  # Exit the loop once a button is clicked

        # Draw the buttons and text
        for i, upgrade in enumerate(upgrade_options):
            button_rect = pygame.Rect(100, 100 + i * 50, 600, 40)
            pygame.draw.rect(screen, BLUE, button_rect)
            text = font.render(upgrade["name"], True, BLACK)
            screen.blit(text, (button_rect.x + 10, button_rect.y + 5))

        pygame.display.flip()
        clock.tick(60)

    # Apply the selected upgrade
    if selected_upgrade == 1:
        player_speed += 1
        print("Player speed increased!")
    elif selected_upgrade == 2:
        bullet_speed += 2
        print("Bullet speed increased!")
    elif selected_upgrade == 10:
        player_size += 10
        player_health += 1  # Increase health when player size increases
        print("Player size increased!")

# Game variables
player = Player(WIDTH // 2, HEIGHT // 2, 50, RED, 5, 3)
enemies = []
enemy_size = 40
enemy_speed = 2

round_number = 1
enemies_defeated = 0
enemies_to_defeat = round_number * 10

bullet_speed = 7
bullet_cooldown = 500  # Cooldown in milliseconds
last_bullet_time = 0

# Game loop
running = True
while running:
    screen.fill(WHITE)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False  # Exit the game when ESC is pressed

    # Player movement
    keys = pygame.key.get_pressed()
    player.move(keys, WIDTH, HEIGHT)

    # Bullet shooting with cooldown
    current_time = pygame.time.get_ticks()
    last_bullet_time = player.shoot(keys, bullet_speed, bullet_cooldown, current_time, last_bullet_time)

    # Move bullets
    for bullet in player.bullets[:]:
        bullet[0] += bullet[2]  # Move horizontally
        bullet[1] += bullet[3]  # Move vertically
        if bullet[0] < 0 or bullet[0] > WIDTH or bullet[1] < 0 or bullet[1] > HEIGHT:
            player.bullets.remove(bullet)  # Remove bullets off-screen

    # Spawn enemies randomly
    if random.randint(1, 50) == 1:  # Adjust this to control enemy spawn rate
        enemies.append(Enemy(random.randint(0, WIDTH - enemy_size), random.randint(0, HEIGHT - enemy_size), enemy_size, GREEN, enemy_speed))

    # Move enemies towards player
    for enemy in enemies:
        enemy.move_towards_player(player.x, player.y)

    # Check bullet-enemy collisions
    for bullet in player.bullets[:]:
        bullet_rect = (bullet[0], bullet[1], 10, 10)
        for enemy in enemies[:]:
            enemy_rect = (enemy.x, enemy.y, enemy.size, enemy.size)
            if check_collision(bullet_rect, enemy_rect):
                player.bullets.remove(bullet)  # Remove bullet
                enemies.remove(enemy)  # Remove enemy
                enemies_defeated += 1

    # Check player-enemy collisions
    player_rect = (player.x, player.y, player.size, player.size)
    for enemy in enemies[:]:
        enemy_rect = (enemy.x, enemy.y, enemy.size, enemy.size)
        if check_collision(player_rect, enemy_rect):
            player.health -= 1  # Decrease player health
            enemies.remove(enemy)  # Remove enemy that hit the player
            if player.health <= 0:
                running = False  # End game if player health is 0

    # Draw player
    player.draw(screen)

    # Draw bullets
    for bullet in player.bullets:
        pygame.draw.rect(screen, BLACK, (bullet[0], bullet[1], 10, 10))

    # Draw enemies
    for enemy in enemies:
        enemy.draw(screen)

    # Display player health and round info
    font = pygame.font.SysFont(None, 36)
    health_text = font.render(f"Health: {player.health}", True, BLACK)
    round_text = font.render(f"Round: {round_number}", True, BLACK)
    screen.blit(health_text, (10, 10))
    screen.blit(round_text, (10, 50))

    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()
sys.exit()
