"""Seaquest game prompt.

https://ale.farama.org/environments/sequest/
"""

GAME_PROMPT: str = """
Sequest Quick Guide:

Goal: You control a submarine to rescue divers while using torpedoes to destroy
enemy submarines and sharks.

Available actions:
- noop: Do nothing.
- fire: Fire.
- up: Move upwards.
- right: Move towards right.
- left: Move towards left.
- down: Move downwards.
- upright: Move diagonally up-right.
- upleft: Move diagonally up-left.
- downright: Move diagonally down-right.
- downleft: Move diagonally down-left.
- upfire: Move upwards and fire.
- rightfire: Move towards right and fire.
- leftfire: Move towards left and fire.
- downfire: Move downwards and fire.
- uprightfire: Move diagonally up-right and fire.
- upleftfire: Move diagonally up-left and fire.
- downrightfire: Move diagonally down-right and fire.
- downleftfire: Move diagonally down-left and fire.

Your Task:

Based on the current state of the game (which will be provided to you), create
your response in the following format:

thought: [Your reasoning about the game state]
move: [action_1, action_2, ..., action_n]

{FPS_SPECIFIC_PROMPT}

Action Planning Guidelines:

- You control a submarine with the goal of rescuing friendly divers. Be aware
  that all other moving objects, including enemy submarines and fish, are
  hostile.
- You must use your torpedoes (fire) to destroy these enemies. Colliding with
  any enemy or the underwater terrain will cost you a life.
- Your submarine has a constantly depleting oxygen supply, and you will also
  lose a life if it runs out.
- You must periodically return to the water's surface to replenish your oxygen.
- Your main goal should be to collect a full load of six divers before 
  surfacing. This is crucial because it increases the point value of all future
  actions, maximizing your scoring potential.
- While surfacing with fewer than six divers results in a penalty (losing one of
  them), it is always the correct choice if the alternative is losing a life
  from lack of oxygen.
- Balance aggression with caution. It's often wise to clear enemies from your
  path to create a safe route for rescuing divers, rather than trying to dodge
  them in a tight spot.
- Scoring is based on shooting enemies and rescuing divers. A significant bonus
  is also awarded for the amount of oxygen remaining when you surface, rewarding
  efficient rescue cycles.
- An extra life (reserve sub) is awarded for every 10,000 points, making
  consistent scoring and survival key to a long game.
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

ACTION_MAP = {
    "noop": 0,
    "fire": 1,
    "up": 2,
    "right": 3,
    "left": 4,
    "down": 5,
    "upright": 6,
    "upleft": 7,
    "downright": 8,
    "downleft": 9,
    "upfire": 10,
    "rightfire": 11,
    "leftfire": 12,
    "downfire": 13,
    "uprightfire": 14,
    "upleftfire": 15,
    "downrightfire": 16,
    "downleftfire": 17,
}

MAX_FRAMES = 500

SKIP_SECONDS = 4.0
