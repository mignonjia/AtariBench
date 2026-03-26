"""Pacman game prompt.

https://ale.farama.org/environments/pacman/
"""

GAME_PROMPT: str = """
Pacman Quick Guide:

Goal: Move Pac Man around a maze collecting food and avoiding ghosts- unless you
eat a Power Pellet, then you can eat the ghosts too!

Available actions:
- noop: Do nothing.
- up: Move up.
- down: Move down.
- left: Move left.
- right: Move right.

Your Task:

Based on the current state of the game (which will be provided to you), create
your response in the following format:

thought: [Your reasoning about the game state]
move: [action_1, action_2, ..., action_n]

{FPS_SPECIFIC_PROMPT}

Action Planning Guidelines:
- You are a Pac Man with a yellow, circular body and a large mouth.
- The small, yellow dashes are the food you need to eat, which you get scores.
- The larger, yellow rectangles are the walls. You can't go through them.
- The pink, ghost-like figures are your enemies. Avoid contact with the ghosts,
  as this will cost you a life. The ghost can jump and run around, so always
  stay away from them.
- The ghosts can split, multiply, and appear or disappear, so historical images
  must be checked to determine their complete positions.
- The larger, flashing pink rectangles are Power Pellets. Eating a Power Pellet
  will temporarily change the color of the ghosts, allowing you to eat them for
  extra points.
- The goal is to clear the entire maze of food to advance to the next level. You
  can use the openings on the sides of the maze to teleport to the opposite
  side.
- The number at the bottom of the screen indicates your current score.
- Plan your route to efficiently collect food while avoiding the ghosts.
"""

FPS_10_PROMPT: str = """
Your plan can contain 1 to 10 actions. Since each action is executed for 0.1
second, your plan is for the next 0.1 to 1 seconds. 

You are encouraged to make shorter plans if the enemy is close or Power Pellets
are near, as shorter plans (fewer actions) will get feedback sooner.

An example of response is:
thought: your thought here
move: [right, right, right]
"""

ACTION_MAP = {
    "noop": 0,
    "up": 1,
    "right": 2,
    "left": 3,
    "down": 4,
}

SKIP_SECONDS = 4.0
MAX_LOST_LIVES = 2
