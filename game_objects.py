import pygame
import random


class Player:
    def __init__(self, x, y, size, color, speed, health, width, height, bullet_size, image_path):
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.speed = speed
        self.health = health
        self.bullets = []
        self.width = width
        self.height = height
        self.bullet_size = bullet_size
        self.move_x = 0
        self.move_y = 0
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (self.size, self.size))
        self.original_image = self.image  # Keep a reference to the original image
        self.angle = 0

    def move(self, keys, width, height):
        prev_x, prev_y = self.x, self.y
        if keys[pygame.K_w] and self.y - self.speed > 0:
            self.y -= self.speed
        if keys[pygame.K_s] and self.y + self.speed + self.size < height:
            self.y += self.speed
        if keys[pygame.K_a] and self.x - self.speed > 0:
            self.x -= self.speed
        if keys[pygame.K_d] and self.x + self.speed + self.size < width:
            self.x += self.speed

        # Calculate the angle of movement
        dx = self.x - prev_x
        dy = self.y - prev_y
        if dx != 0 or dy != 0:
            self.angle = (180 / 3.14159) * -pygame.math.Vector2(dx, dy).angle_to((1, 0))
            self.image = pygame.transform.rotate(self.original_image, self.angle)

    def avoid_enemies(self, enemies, width, height):
        """AI function to move the player smoothly away from enemies while seeking open spaces."""
        safe_distance = 175  # Distance at which enemies start to repel the player
        move_x, move_y = 0, 0

        # Calculate repulsive forces from enemies
        for enemy in enemies:
            distance_x = self.x - enemy.x
            distance_y = self.y - enemy.y
            distance = (distance_x ** 2 + distance_y ** 2) ** 0.5

            if distance < safe_distance:
                repulsion_strength = safe_distance / (distance ** 2)
                move_x += distance_x * repulsion_strength
                move_y += distance_y * repulsion_strength

        # Calculate attractive force towards the center of open space
        center_x, center_y = width / 2, height / 2
        attract_x = (center_x - self.x) * 0.01  # Small factor for smooth attraction
        attract_y = (center_y - self.y) * 0.01  # Small factor for smooth attraction

        # Combine forces
        move_x += attract_x
        move_y += attract_y

        norm = (move_x ** 2 + move_y ** 2) ** 0.5
        if norm != 0:
            move_x = (move_x / norm) * self.speed
            move_y = (move_y / norm) * self.speed

        # Smooth movement
        smoothing_factor = 0.3
        self.move_x = (1 - smoothing_factor) * self.move_x + smoothing_factor * move_x
        self.move_y = (1 - smoothing_factor) * self.move_y + smoothing_factor * move_y

        self.x += self.move_x
        self.y += self.move_y

        # Constrain movement within screen boundaries
        self.x = max(0, min(width - self.size, self.x))
        self.y = max(0, min(height - self.size, self.y))

    def avoid_enemies2(self, enemies, width, height):
        """AI function to move the player smoothly away from enemies while seeking open spaces."""
        safe_distance = 175  # Distance at which enemies start to repel the player
        move_x, move_y = 0, 0

        # Calculate repulsive forces from enemies
        for enemy, information in enemies.items():
            distance_x = self.x - information["x"]
            distance_y = self.y - information["y"]
            distance = (distance_x ** 2 + distance_y ** 2) ** 0.5

            if distance < safe_distance:
                repulsion_strength = safe_distance / (distance ** 2)
                move_x += distance_x * repulsion_strength
                move_y += distance_y * repulsion_strength

        # Calculate attractive force towards the center of open space
        center_x, center_y = width / 2, height / 2
        attract_x = (center_x - self.x) * 0.01  # Small factor for smooth attraction
        attract_y = (center_y - self.y) * 0.01  # Small factor for smooth attraction

        # Combine forces
        move_x += attract_x
        move_y += attract_y

        norm = (move_x ** 2 + move_y ** 2) ** 0.5
        if norm != 0:
            move_x = (move_x / norm) * self.speed
            move_y = (move_y / norm) * self.speed

        # Smooth movement
        smoothing_factor = 0.3
        self.move_x = (1 - smoothing_factor) * self.move_x + smoothing_factor * move_x
        self.move_y = (1 - smoothing_factor) * self.move_y + smoothing_factor * move_y

        self.x += self.move_x
        self.y += self.move_y

        # Constrain movement within screen boundaries
        self.x = max(0, min(width - self.size, self.x))
        self.y = max(0, min(height - self.size, self.y))

    def shoot(self, keys, bullet_speed, bullet_cooldown, current_time, last_bullet_time):
        if current_time - last_bullet_time >= bullet_cooldown:
            bullet_width = self.bullet_size

            if keys[pygame.K_UP]:
                self.bullets.append(
                    [self.x + self.size // 3, self.y, 0, -bullet_speed, self.x, self.y, bullet_width])
            elif keys[pygame.K_DOWN]:
                self.bullets.append(
                    [self.x + self.size // 3, self.y + self.size, 0, bullet_speed, self.x, self.y,
                     bullet_width])
            elif keys[pygame.K_LEFT]:
                self.bullets.append(
                    [self.x, self.y + self.size // 3, -bullet_speed, 0, self.x, self.y, bullet_width])
            elif keys[pygame.K_RIGHT]:
                self.bullets.append(
                    [self.x + self.size, self.y + self.size // 3, bullet_speed, 0, self.x, self.y,
                     bullet_width])

            last_bullet_time = current_time

        return last_bullet_time

    def draw(self, screen):
        rotated_rect = self.image.get_rect(center=(self.x + self.size // 2, self.y + self.size // 2))
        screen.blit(self.image, rotated_rect.topleft)


class Enemy:
    def __init__(self, x, y, size, color, speed, image_path):
        self.x = x
        self.y = y
        self.size = size
        self.color = color
        self.speed = speed
        self.velocity = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        self.image = pygame.image.load(image_path)
        self.image = pygame.transform.scale(self.image, (self.size, self.size))
        self.original_image = self.image  # Keep a reference to the original image
        self.angle = 0

    def update(self, new_x, new_y):
        # Update the stored position based on the model's decision
        self.x = new_x
        self.y = new_y

    def move_towards_player(self, player_x, player_y, enemies, separation_distance=100, cohesion_factor=0.005,
                            alignment_factor=0.05, target_factor=0.1):

        separation = pygame.Vector2(0, 0)
        count = 0
        for other in enemies:
            if other != self:
                dist = pygame.Vector2(self.x - other.x, self.y - other.y).length()
                if dist < separation_distance:
                    repulsion_force = 1 / dist
                    separation += (pygame.Vector2(self.x - other.x, self.y - other.y).normalize() * repulsion_force)
                    count += 1
        if count > 0:
            separation /= count

        alignment = pygame.Vector2(0, 0)
        count = 0
        for other in enemies:
            if other != self:
                dist = pygame.Vector2(self.x - other.x, self.y - other.y).length()
                if dist < separation_distance:
                    alignment += other.velocity
                    count += 1
        if count > 0:
            alignment = (alignment / count).normalize() * alignment_factor

        cohesion_vector = pygame.Vector2(player_x - self.x, player_y - self.y).normalize()
        distance_to_player = cohesion_vector.length()

        if distance_to_player < separation_distance:
            cohesion_factor *= (distance_to_player / separation_distance)

        cohesion = cohesion_vector * cohesion_factor

        target_direction = pygame.Vector2(player_x - self.x, player_y - self.y).normalize() * target_factor

        self.velocity += separation + alignment + cohesion + target_direction
        if self.velocity.length() > self.speed:
            self.velocity = self.velocity.normalize() * self.speed

        self.x += self.velocity.x
        self.y += self.velocity.y

        direction_vector = pygame.Vector2(player_x - self.x, player_y - self.y)

        # Normalize direction and move the enemy
        if direction_vector.length() > 0:
            direction_vector = direction_vector.normalize() * self.speed
            self.x += direction_vector.x
            self.y += direction_vector.y

        self.angle = (180 / 3.14159) * -direction_vector.angle_to((1, 0))
        self.image = pygame.transform.rotate(self.original_image, self.angle)

    def draw(self, screen):
        rotated_rect = self.image.get_rect(center=(self.x + self.size // 2, self.y + self.size // 2))
        screen.blit(self.image, rotated_rect.topleft)
