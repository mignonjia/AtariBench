"""Assault game prompt.

https://ale.farama.org/environments/assault/
"""

GAME_PROMPT: str = """
Assault Quick Guide:

Goal: Points are earned by destroying enemy ships with your cannon.

Available actions:
- noop: Do nothing. The cannon remains stationary.
- fire up: Fire up.
- move left: Move the cannon to the left.
- move right: Move the cannon to the right.
- fire left: Fire to the left.
- fire right: Fire to the right.

Your Task:

Based on the current state of the game (which will be provided to you), create
your response in the following format:

thought: [Your reasoning about the game state]
move: [action_1, action_2, ..., action_n]

{FPS_SPECIFIC_PROMPT}

Action Planning Guidelines:
- Your primary objective is to shoot down the ten alien ships deployed by the
  mothership in each wave.
- Enemy projectiles, including fireballs, can attack your cannon horizontally.
  You can move your cannon left and right to dodge them. If you got hit by an
  enemy projectile, you will lose a life.
- The mothership itself cannot be destroyed.
- Alien ships exhibit different movement patterns depending on the wave,
  including horizontal movement with sudden direction changes, combined up-down
  and horizontal movement.
- Important: Be aware of the cannon's temperature, indicated by a bar in the
  bottom right. The length of the bar indicates the temperature. Continuous
  firing will cause it to overheat, resulting in the loss of a cannon. Make sure
  the cannon is not overheating (the green bar is not too long or even turn to
  red).
- Firing left or right when an enemy is throwed to your platform.
- After a move left or right command, expect the cannon to continue moving
  slightly for an additional 0.1 seconds no matter what next action you choose.
"""


FPS_10_PROMPT: str = """
Your plan can contain 1 to 10 actions. Since each action is executed for 0.1
second, your plan is for the next 0.1 to 1 seconds.

You are encouraged to make shorter plans, as shorter plans (fewer actions) will
get feedback sooner. This can give you more recent game states so that you can
take better actions.

An example of response is:
thought: your thought here
move: [move left, move left]
"""


ACTION_MAP = {
    "noop": 0,
    "fire up": 2,
    "move right": 3,
    "move left": 4,
    "fire right": 5,
    "fire left": 6,
}

MAX_LOST_LIVES = 1
