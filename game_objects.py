import pygame
import math
import random

class Player:
    def __init__(self, x, y, size, color, speed, health, width, height):
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.speed = speed
        self.health = health
        self.bullets = []
        self.width = width
        self.height = height

    def move(self, keys, width, height):
        if keys[pygame.K_w] and self.y - self.speed > 0:
            self.y -= self.speed
        if keys[pygame.K_s] and self.y + self.speed + self.size < height:
            self.y += self.speed
        if keys[pygame.K_a] and self.x - self.speed > 0:
            self.x -= self.speed
        if keys[pygame.K_d] and self.x + self.speed + self.size < width:
            self.x += self.speed
    '''
    def avoid_enemies(player, enemies):
        """AI function to move the player smoothly away from nearby enemies while seeking open spaces."""
        safe_distance = 250  # Minimum distance to maintain from enemies
        move_x, move_y = 0, 0

        for enemy in enemies:
            distance_x = player.x - enemy.x
            distance_y = player.y - enemy.y
            distance = (distance_x ** 2 + distance_y ** 2) ** 0.5

            if distance < safe_distance:
                move_x += distance_x / (distance ** 2)
                move_y += distance_y / (distance ** 2)

        if move_x != 0 or move_y != 0:
            norm = (move_x ** 2 + move_y ** 2) ** 0.5
            move_x = (move_x / norm) * player.speed
            move_y = (move_y / norm) * player.speed

        if player.x + move_x < 0 or player.x + move_x > player.width - player.size:
            move_x = 0  # Prevent moving outside horizontal boundaries
        if player.y + move_y < 0 or player.y + move_y > player.height - player.size:
            move_y = 0  # Prevent moving outside vertical boundaries

        if player.x <= 0 or player.x >= player.width - player.size:
            move_x = player.speed  # Attempt to move away from the wall horizontally
        if player.y <= 0 or player.y >= player.height - player.size:
            move_y = player.speed  # Attempt to move away from the wall vertically

        player.x += move_x
        player.y += move_y

        player.x = max(0, min(player.width - player.size, player.x))
        player.y = max(0, min(player.height - player.size, player.y))
    '''
    def shoot(self, keys, bullet_speed, bullet_cooldown, current_time, last_bullet_time):
        if current_time - last_bullet_time >= bullet_cooldown:
            if keys[pygame.K_UP]:
                self.bullets.append([self.x + self.size // 3, self.y, 0, -bullet_speed, self.x, self.y])  # Up
            elif keys[pygame.K_DOWN]:
                self.bullets.append(
                    [self.x + self.size // 3, self.y + self.size, 0, bullet_speed, self.x, self.y])  # Down
            elif keys[pygame.K_LEFT]:
                self.bullets.append([self.x, self.y + self.size // 3, -bullet_speed, 0, self.x, self.y])  # Left
            elif keys[pygame.K_RIGHT]:
                self.bullets.append(
                    [self.x + self.size, self.y + self.size // 3, bullet_speed, 0, self.x, self.y])  # Right
            last_bullet_time = current_time
        return last_bullet_time

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.size, self.size))

class Enemy:
    def __init__(self, x, y, size, color, speed):
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.speed = speed

    def move_towards_player(self, player_x, player_y):
        angle = math.atan2(player_y - self.y, player_x - self.x)
        self.x += self.speed * math.cos(angle)
        self.y += self.speed * math.sin(angle)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.size, self.size))
