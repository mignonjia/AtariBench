"""Freeway game prompt.

https://ale.farama.org/environments/freeway/
"""

GAME_PROMPT: str = """
Freeway Quick Guide:

Goal: Your objective is to guide your chicken across lane after lane of busy
rush hour traffic. You receive a point for every chicken that makes it to the
top of the screen after crossing all the lanes of traffic.

Available actions:
- noop: Do nothing. The paddle remains stationary.
- up: Move the chicken up.
- down: Move the chicken down.

Your Task:

Based on the current state of the game (which will be provided to you), create
your response in the following format:

thought: [Your reasoning about the game state]
move: [action_1, action_2, ..., action_n]

{FPS_SPECIFIC_PROMPT}

Action Planning Guidelines:
- You have 50 seconds in total. The final reward is determined by how many
  points you can earn.
- There are two chickens in the game. You can only control the left chicken in
  the current game in this single player mode.
- Pay attention to the direction and speed of the traffic, so the chicken
  can avoid the cars. If a collision happens, the chicken will be forced to move
  down for 2 lanes.
- Prioritize moving up. If there will be a collusion in the next lane, noop to
  avoid it. If the current lane and the next lane are both in danger, move
  down is also a option.
"""

FPS_10_PROMPT: str = """
Your plan can contain 1 to 10 actions. Since each action is executed for 0.1
second, your plan is for the next 0.1 to 1 seconds.

You are encouraged to make shorter plans if the car is close, so you can closely
monitor the states and make better next steps.

An example of response is:
thought: your thought here
move: [up, up, up]
"""

ACTION_MAP = {
    "noop": 0,
    "up": 1,
    "down": 2,
}

MAX_FRAMES = 500
