"""DemonAttack game prompt.

https://ale.farama.org/environments/demon_attack/
"""

GAME_PROMPT: str = """
DemonAttack Quick Guide:

Goal: You control a spaceship that can move sideways. You are facing waves of
demons in the ice planet of Krybor. Points are accumulated by destroying demons.
Every time an enemy hits you, you will lose a life. Your goal is to survive and
earn as many points as possible.

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
- You control the spaceship at the bottom of the screen, which can move
  horizontally.
- Your enemies are demons flying in the air, and they will fire lasers
  downwards to attack you. You need to move horizontally to avoid being hit by
  lasers.
- Provided you won't be hit, you can shoot down the demons in the air as much as
  possible. The specific method is to move left or right to directly below them,
  and then fire. It is especially important to note that avoiding lasers has a
  higher priority than shooting down demons. That is to say, if there is a laser
  where you are going to move, avoid the laser first, and then shoot down the 
  demon.
"""

FPS_10_PROMPT: str = """
Your plan can contain 1 to 10 actions. Since each action is executed for 0.1
second, your plan is for the next 0.1 to 1 seconds.

You are encouraged to make shorter plans, as shorter plans (fewer actions) will
get feedback sooner. This can give you more recent game states so that you can
take better actions.

An example of response is:
thought: your thought here
move: [right, left, fire]
"""


ACTION_MAP = {
    "noop": 0,
    "fire": 1,
    "right": 2,
    "left": 3,
    "rightfire": 4,
    "leftfire": 5,
}

MAX_LOST_LIVES = 2
