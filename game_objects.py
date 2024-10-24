import pygame
import math

class Player:
    def __init__(self, x, y, size, color, speed, health):
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.speed = speed
        self.health = health
        self.bullets = []

    def move(self, keys, width, height):
        if keys[pygame.K_w] and self.y - self.speed > 0:
            self.y -= self.speed
        if keys[pygame.K_s] and self.y + self.speed + self.size < height:
            self.y += self.speed
        if keys[pygame.K_a] and self.x - self.speed > 0:
            self.x -= self.speed
        if keys[pygame.K_d] and self.x + self.speed + self.size < width:
            self.x += self.speed

    def shoot(self, keys, bullet_speed, bullet_cooldown, current_time, last_bullet_time):
        if current_time - last_bullet_time >= bullet_cooldown:
            if keys[pygame.K_UP]:
                self.bullets.append([self.x + self.size // 2, self.y, 0, -bullet_speed, self.x, self.y])  # Up
            elif keys[pygame.K_DOWN]:
                self.bullets.append(
                    [self.x + self.size // 2, self.y + self.size, 0, bullet_speed, self.x, self.y])  # Down
            elif keys[pygame.K_LEFT]:
                self.bullets.append([self.x, self.y + self.size // 2, -bullet_speed, 0, self.x, self.y])  # Left
            elif keys[pygame.K_RIGHT]:
                self.bullets.append(
                    [self.x + self.size, self.y + self.size // 2, bullet_speed, 0, self.x, self.y])  # Right
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
