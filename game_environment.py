import pygame
import random
from game_objects import Player, Enemy


class GameEnv:
    def __init__(self, width, height):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.player = Player(width // 2, height // 2, 30, (255, 0, 0), 4, 1, width, height, 5)
        self.enemies = []
        self.spawn_interval = 600
        self.last_spawn_time = 0
        self.start_time = pygame.time.get_ticks()
        self.max_duration = 30000
        self.max_enemies = 25
        self.done = False

    def reset(self):
        self.player.x, self.player.y = self.width // 2, self.height // 2
        self.player.health = 1
        self.enemies.clear()
        self.last_spawn_time = pygame.time.get_ticks()
        self.start_time = pygame.time.get_ticks()
        self.done = False
        return self.get_state()

    def get_state(self):
        player_state = [self.player.x, self.player.y]
        enemies_state = []

        for enemy in self.enemies:
            enemies_state.extend([enemy.x, enemy.y, enemy.size])

        # Ensure a fixed length state even if there are no enemies
        max_enemies = self.max_enemies
        while len(enemies_state) < max_enemies * 3:
            enemies_state.extend([0, 0, 0])  # Padding with zeros

        return player_state + enemies_state

    def step(self, action):
        self.handle_input(action)

        current_time = pygame.time.get_ticks()
        if current_time - self.last_spawn_time > self.spawn_interval:
            self.spawn_enemy()
            self.last_spawn_time = current_time

        self.update_game_objects()
        reward, self.done = self.check_collisions()
        elapsed_time = current_time - self.start_time

        if elapsed_time >= self.max_duration:
            reward += 10
            self.done = True

        state = self.get_state()
        return state, reward, self.done

    def handle_input(self, action):
        if action == "UP":
            self.player.y = max(0, self.player.y - self.player.speed)
        elif action == "DOWN":
            self.player.y = min(self.height - self.player.size, self.player.y + self.player.speed)
        elif action == "LEFT":
            self.player.x = max(0, self.player.x - self.player.speed)
        elif action == "RIGHT":
            self.player.x = min(self.width - self.player.size, self.player.x + self.player.speed)

    def spawn_enemy(self):
        """Function to spawn enemies at a given location."""
        enemy_information = {
            'yellow': [30, 2.6, (255, 255, 0)],  # [Size, speed, color]
            'green': [15, 3.0, (0, 255, 0)],
            'red': [60, 2, (255, 0, 0)],
            'purple': [40, 3.3, (255, 0, 255)],
        }

        if len(self.enemies) >= self.max_enemies:
            return None

        edge = random.choice(['top', 'bottom', 'left', 'right'])

        # Always choose 'yellow' enemy for the environment
        enemy_color = 'yellow'
        enemy_size = enemy_information[enemy_color][0]
        enemy_speed = enemy_information[enemy_color][1]
        enemy_color = enemy_information[enemy_color][2]

        x, y = 0, 0

        if edge == 'top':
            x = random.randint(0, self.width - enemy_size)
            y = 0
        elif edge == 'bottom':
            x = random.randint(0, self.width - enemy_size)
            y = self.height - enemy_size
        elif edge == 'left':
            x = 0
            y = random.randint(0, self.height - enemy_size)
        elif edge == 'right':
            x = self.width - enemy_size
            y = random.randint(0, self.height - enemy_size)

        new_enemy = Enemy(x, y, enemy_size, enemy_color, enemy_speed)
        self.enemies.append(new_enemy)

    def update_game_objects(self):
        for enemy in self.enemies:
            enemy.move_towards_player(self.player.x, self.player.y, self.enemies)

    def check_collisions(self):
        player_rect = (self.player.x, self.player.y, self.player.size, self.player.size)
        for enemy in self.enemies:
            enemy_rect = (enemy.x, enemy.y, enemy.size, enemy.size)
            if pygame.Rect(player_rect).colliderect(pygame.Rect(enemy_rect)):
                return -10, True
        return 0, False


    def render(self):
        self.screen.fill((230, 230, 230))
        self.player.draw(self.screen)
        for enemy in self.enemies:
            enemy.draw(self.screen)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()


def main():
    game_env = GameEnv(800, 600)

    while not game_env.done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_env.close()
                return

        keys = pygame.key.get_pressed()
        action = None
        if keys[pygame.K_UP]:
            action = "UP"
        elif keys[pygame.K_DOWN]:
            action = "DOWN"
        elif keys[pygame.K_LEFT]:
            action = "LEFT"
        elif keys[pygame.K_RIGHT]:
            action = "RIGHT"

        if action:
            game_env.step(action)
            print(game_env.get_state())

        game_env.render()

    print("Game Over!")
    pygame.time.delay(2000)
    game_env.close()


if __name__ == "__main__":
    main()