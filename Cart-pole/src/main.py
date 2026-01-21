import numpy as np
import pygame
import pickle

from QAgent import QLearningAgent
from cartpole import CartPole
from config import *

def world_to_screen(x, y):
    sx = WIDTH // 2 + int(x * SCALE)
    sy = HEIGHT // 2 - int(y * SCALE)
    return sx, sy

if __name__ == "__main__":

    CP = CartPole()
    agent = QLearningAgent()

    # Pygame init
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    # Training mode
    training = True
    episode = 0
    episode_steps = 0
    best_steps = 0
    recent_scores = []
    avg_score = 0
    
    running = True
    fast_mode = False
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    training = not training
                    print(f"Training mode: {training}")
                elif event.key == pygame.K_s:
                    with open('q_table.pkl', 'wb') as f:
                        pickle.dump(agent.q_table, f)
                    print("Q-table saved!")
                elif event.key == pygame.K_l:
                    try:
                        with open('q_table.pkl', 'rb') as f:
                            agent.q_table = pickle.load(f)
                        print("Q-table loaded!")
                    except:
                        print("No saved Q-table found")
                elif event.key == pygame.K_f:
                    fast_mode = not fast_mode
                    print(f"Fast mode: {fast_mode}")
                elif event.key == pygame.K_r:
                    # Reset Q-table
                    agent.q_table = np.random.uniform(-0.01, 0.01, (X_BINS, THETA_BINS, X_DOT_BINS, THETA_DOT_BINS, N_ACTIONS))
                    agent.epsilon = EPSILON_START
                    episode = 0
                    best_steps = 0
                    recent_scores = []
                    print("Agent reset!")

        # Get current state
        state = CP.discretize_state()
        
        # Choose action
        action_idx = agent.get_action(state, training=training)
        u = ACTIONS[action_idx]
        
        # Take step
        z_prev = CP.z.copy()
        CP.z, cart_xy, pole_xy = CP.step(u)
        next_state = CP.discretize_state()
        
        # Check if terminated
        terminated = CP.is_terminal()
        
        # Get reward
        reward = CP.get_reward(z_prev, terminated)
        
        # Update Q-table if training
        if training:
            agent.update(state, action_idx, reward, next_state, terminated)
        
        episode_steps += 1
        
        # Reset if terminated
        if terminated:
            if episode_steps > best_steps:
                best_steps = episode_steps
            
            recent_scores.append(episode_steps)
            if len(recent_scores) > 100:
                recent_scores.pop(0)
            avg_score = sum(recent_scores) / len(recent_scores)
            
            episode += 1
            episode_steps = 0
            
            # Random initial angle
            CP.reset()
            
            if training:
                agent.decay_epsilon()
            
            if episode % 50 == 0:
                print(f"Episode {episode}, Best: {best_steps}, Avg(100): {avg_score:.1f}, Epsilon: {agent.epsilon:.4f}")

        # Drawing (skip if in fast mode during training)
        if not fast_mode or not training:
            cart_sx, cart_sy = world_to_screen(cart_xy[0], cart_xy[1])
            pole_sx, pole_sy = world_to_screen(pole_xy[0], pole_xy[1])

            screen.fill((255, 255, 255))
            
            # Draw track
            track_y = HEIGHT // 2
            pygame.draw.line(screen, (100, 100, 100), (0, track_y), (WIDTH, track_y), 2)

            # Draw cart
            cart_rect = pygame.Rect(0, 0, CART_W, CART_H)
            cart_rect.center = (cart_sx, cart_sy)
            pygame.draw.rect(screen, (0, 0, 0), cart_rect)

            # Draw pole
            pygame.draw.line(screen, (200, 0, 0), (cart_sx, cart_sy), (pole_sx, pole_sy), 4)
            pygame.draw.circle(screen, (0, 0, 200), (pole_sx, pole_sy), 6)

            # Display info
            mode_text = "TRAINING" if training else "TESTING"
            mode_color = (0, 150, 0) if training else (0, 0, 200)
            
            text1 = font.render(f"{mode_text}", True, mode_color)
            text2 = small_font.render(f"Episode: {episode}  Steps: {episode_steps}  Best: {best_steps}  Avg: {avg_score:.1f}", True, (0, 0, 0))
            text3 = small_font.render(f"Epsilon: {agent.epsilon:.4f}  Angle: {np.rad2deg(CP.z[1]):.1f}°  Action: {u:.0f}", True, (0, 0, 0))
            text4 = small_font.render(f"T: Toggle  S: Save  L: Load  F: Fast  R: Reset", True, (100, 100, 100))
            
            screen.blit(text1, (10, 10))
            screen.blit(text2, (10, 50))
            screen.blit(text3, (10, 75))
            screen.blit(text4, (10, HEIGHT - 30))

            pygame.display.flip()
            clock.tick(int(1/CP.dt))

    pygame.quit()