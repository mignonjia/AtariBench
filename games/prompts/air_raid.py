"""AirRaid game prompt.

https://ale.farama.org/environments/air_raid/
"""

GAME_PROMPT: str = """
AirRaid Quick Guide:

Goal: You control a ship that can move sideways. Your will earn scores by
shooting down the flying saucers that are trying to drop bombs.

Available actions:
- noop: Do nothing.
- fire: Fire up.
- right: Move right.
- left: Move left.
- rightfire: Move right and fire up.
- leftfire: Move left and fire up.

Your Task:

Based on the current state of the game (which will be provided to you), create
your response in the following format:

thought: [Your reasoning about the game state]
move: [action_1, action_2, ..., action_n]

{FPS_SPECIFIC_PROMPT}

Action Planning Guidelines:

- You control a ship that can move sideways. Your will earn scores by
  shooting down the flying saucers that are trying to drop bombs.
- Enemies can only move vertically downwards, so if you want to shoot them down,
  you just need to move directly below them and then fire up.
- Note that there is a minimum time interval between fires, so you cannot keep
  firing.
- You will receive score for each enemy you destroy. So destroy as many enemies
  as possible.

- At the same time, enemy saucers can also hit you by firing downwards,
  so while you are trying to attack enemy saucers, you also need to pay
  attention to dodge horizontally to avoid being hit by their bombs.
- The small ships at the bottom of the screen indicate your remaining lives.
"""

FPS_10_PROMPT: str = """
Your plan can contain 1 to 10 actions. Since each action is executed for 0.1
second, your plan is for the next 0.1 to 1 seconds.

You are encouraged to make shorter plans, as shorter plans (fewer actions) will
get feedback sooner. This can give you more recent game states so that you can
take better actions.

An example of response is:
thought: your thought here
move: [right, right, right]
"""

ACTION_MAP = {
    "noop": 0,
    "fire": 1,
    "right": 2,
    "left": 3,
    "rightfire": 4,
    "leftfire": 5,
}

# Although there are 3 lives as shown in the image, it is not reflected in the
# info or rewards. So we cannot use max_lost_lives to terminate the game.
# Instead, we use max_frames to terminate the game.

MAX_FRAMES = 500
