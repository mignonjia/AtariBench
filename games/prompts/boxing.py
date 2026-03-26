"""Boxing game prompt.

https://ale.farama.org/environments/boxing/
"""

GAME_PROMPT: str = """
Boxing Quick Guide:

Goal: You fight an opponent in a boxing ring. You score points for hitting the
opponent, and your opponent scores points for hitting you. Your goal is to score
as many points as possible within 50 seconds, while let your opponent score as
few points as possible.

Available actions:
- noop: Do nothing.
- punch: punch.
- up: move up
- right: move right
- left: move left
- down: move down
- upright: move up and right
- upleft: move up and left
- downright: move down and right
- downleft: move down and left
- uppunch: move up and punch
- rightpunch: move right and punch
- leftpunch: move left and punch
- downpunch: move down and punch
- uprightpunch: move up and right and punch
- upleftpunch: move up and left and punch
- downrightpunch: move down and right and punch
- downleftpunch: move down and left and punch

Your Task:

Based on the current state of the game (which will be provided to you), create
your response in the following format:

thought: [Your reasoning about the game state]
move: [action_1, action_2, ..., action_n]

{FPS_SPECIFIC_PROMPT}

Action Planning Guidelines:
- You control the white boxer. His head is in the middle, and his gloved hands
  are above and below. He can move up, down, left, and right within the area
  defined by the ropes.
- Your enemy is the black boxer. Similarly, his head is in the middle, and his
  gloved hands are above and below. He can move up, down, left, and right within
  the area defined by the ropes.
- When you punch, the white boxer will throw either the left or right hand,
  depending on the enemy's position. The hand closer to the enemy will extend
  horizontally.
- You score points when your thrown fist hits the enemy's head. Therefore, you
  should try to keep your fist and the enemy's head at the same horizontal level
  and close enough, then throw a punch. Similarly, if your head and the enemy's
  fist are at the same horizontal level, you should try to stay away from the
  enemy.
- An effective way to score is to try to push the enemy against the ropes and
  then punch wildly, which will give you more opportunities to hit the enemy and
  score.
"""

FPS_10_PROMPT: str = """
Your plan can contain 1 to 10 actions. Since each action is executed for 0.1
second, your plan is for the next 0.1 to 1 seconds.

You are encouraged to make shorter plans, as shorter plans (fewer actions) will
get feedback sooner. This can give you more recent game states so that you can
take better actions.

An example of response is:
thought: your thought here
move: [up, right, punch]
"""


ACTION_MAP = {
    "noop": 0,
    "punch": 1,
    "up": 2,
    "right": 3,
    "left": 4,
    "down": 5,
    "upright": 6,
    "upleft": 7,
    "downright": 8,
    "downleft": 9,
    "uppunch": 10,
    "rightpunch": 11,
    "leftpunch": 12,
    "downpunch": 13,
    "uprightpunch": 14,
    "upleftpunch": 15,
    "downrightpunch": 16,
    "downleftpunch": 17,
}

MAX_FRAMES = 500
