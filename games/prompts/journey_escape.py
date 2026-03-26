"""JourneyEscape game prompt.

https://ale.farama.org/environments/journey_escape/
"""

GAME_PROMPT: str = """
Journey Escape Quick Guide:

Goal: Your goal is to avoid the descending groupies, photographers, and
promoters. If an obstacle moving downwards collides with you from above, you
will lose 1 point. While you don't need to move up to progress, you can move
upwards to avoid obstacles.

Available actions:
- noop: Do nothing.
- up: Move up.
- right: Move right.
- left: Move left.
- down: Move down.
- upright: Move up and right.
- upleft: Move up and left.
- downright: Move down and right.
- downleft: Move down and left.

For all the moving directions, you can speed up the move by adding "fast" before
the direction, including "fastright", "fastleft", "fastdown", "fastupright",
"fastupleft", "fastdownright", "fastdownleft".
Note that "noop" and "up" don't have "fast" version, which means "fastnoop" and
"fastup" are the same as "noop" and "up".

Your Task:

Based on the current state of the game (which will be provided to you), create
your response in the following format:

thought: [Your reasoning about the game state]
move: [action_1, action_2, ..., action_n]

{FPS_SPECIFIC_PROMPT}

Action Planning Guidelines:
- You are optimizing score over the fixed game window, so avoid unnecessary
  point losses and keep moving through safer lanes.

- Enemies & Obstacles (Avoid, 1 point loss per encounter):
  - Shifty-Eyed Promoters (unhappy torsos).
  - Sneaky Photographers (flashing crystal balls).
  - Love-Crazed Groupies (hearts with legs).
  - Stage Barriers (brick walls).

- Helpful Characters (Seek):
  - Loyal Roadie (blue alien): Touching this character grants temporary
    invincibility, allowing you to move through all obstacles without penalty.
  - Mighty Manager (looks like the Kool-Aid Man): This character is rare.
    Contacting him gives you a 1-point bonus and provides a clear, unstoppable
    path directly to the escape vehicle.

- Ignore the score (money) at the top center of the screen, as I will use the
  above scoring details to calculate the score instead of the original game
  score.
"""

FPS_10_PROMPT: str = """
Your plan can contain 1 to 10 actions. Since each action is executed for 0.1
second, your plan is for the next 0.1 to 1 seconds.

You are encouraged to make shorter plans if the obstacles are close, as shorter
plans (fewer actions) will get feedback sooner.

An example of response is:
thought: your thought here
move: [right, right, right]
"""

ACTION_MAP = {
    "noop": 0,
    "fastnoop": 0,
    "up": 1,
    "fastup": 1,
    "right": 2,
    "left": 3,
    "down": 4,
    "upright": 5,
    "upleft": 6,
    "downright": 7,
    "downleft": 8,
    "fastright": 9,
    "fastleft": 10,
    "fastdown": 11,
    "fastupright": 11,
    "fastupleft": 12,
    "fastdownright": 13,
    "fastdownleft": 14,
}

MAX_FRAMES = 500
