"""TimePilot game prompt.

https://ale.farama.org/environments/timepilot/
"""

GAME_PROMPT: str = """
TimePilot Quick Guide:

Goal: You control an aircraft. Use it to destroy your enemies.

Available actions:
- noop: Do nothing.
- fire: Fire.
- up: Move upwards.
- right: Move towards right.
- left: Move towards left.
- down: Move downwards.

Your Task:

Based on the current state of the game (which will be provided to you), create
your response in the following format:

thought: [Your reasoning about the game state]
move: [action_1, action_2, ..., action_n]

{FPS_SPECIFIC_PROMPT}

Action Planning Guidelines:

- You control an aircraft that can move in 8 directions (up, down, right, left,
  up-right, down-right, up-left, down-left).
- Your goal is to defeat as many enemies as possible while avoiding being hit by
  enemy bullets. Your bullets are fired in the direction you are moving, so you
  can anticipate when an enemy will appear in your path and start firing, or you
  can follow behind an enemy and fire forwards.
- You will receive a score for each enemy you destroy, so destroy as many
  enemies as possible.
"""

FPS_10_PROMPT: str = """
Your plan can contain 1 to 10 actions. Since each action is executed for 0.1
second, your plan is for the next 0.1 to 1 seconds.

You are encouraged to make shorter plans, as shorter plans (fewer actions) will
get feedback sooner. This can give you more recent game states so that you can
take better actions.

An example of response is:
thought: your thought here
move: [right, right, fire]
"""

ACTION_MAP = {"noop": 0, "fire": 1, "up": 2, "right": 3, "left": 4, "down": 5}

MAX_FRAMES = 500

SKIP_SECONDS = 4.0
