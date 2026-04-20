"""Qbert game prompt.

https://ale.farama.org/environments/qbert/
"""

GAME_PROMPT: str = """
Qbert Quick Guide:

Goal: Change the color of all cubes on the pyramid to the destination color
while avoiding jumping off the pyramid, getting caught by the snake, and getting
hit by the red ball.

Available actions:
- noop: Do nothing.
- top-right: Hop to the top-right cube.
- bottom-right: Hop to the bottom-right cube.
- top-left: Hop to the top-left cube.
- bottom-left: Hop to the bottom-left cube.

Your Task:

Based on the current state of the game (which will be provided to you), create
your response in the following format:

thought: [Your reasoning about the game state]
move: [action_1, action_2, ..., action_n]

{FPS_SPECIFIC_PROMPT}

Action Planning Guidelines:

- Press any action to start the game from the initial screen.
- Your primary objective is to change every cube's color on the pyramid. You
  start with 2 lives.
- Jumping on a cube changes its color. The rules for color changes become more
  complex in later levels.
- You can jump to any cube with the corresponding action. But if you jump to a
  direction that is not a cube, you will lose a life. Losing a life does not
  directly reduce score, but the recovery consumes time, so only jump in
  directions that actually contain a cube.
- Survival is critical. You lose a life if you jump off the pyramid or get
  caught by Coily (the snake) or a Red Ball.
- Enemy Guide:
    - Red Ball: Drops from the top. Avoid its path.
    - Coily (Snake): Hatches from a Purple Ball at the bottom and pursues you.
      The highest priority is to evade Coily.
    - Sam (Green creature): Changes cubes back to their previous color. Running
      into him scores bonus points.
    - Green Ball: A helpful item. Touching it temporarily freezes all enemies
      and scores bonus points.
- Flying Saucers: These are your escape route. Lure Coily to the edge of the
  pyramid and jump onto a saucer just as he is about to catch you. He will jump
  off the pyramid, and you will earn a large bonus (500 points). You will be
  safely transported back to the top of the pyramid.
- Scoring: You get points for changing cubes (25), catching Sam (300), catching
  a green ball (100), luring Coily off the pyramid (500), and completing a round
  (bonus). You earn an extra life for every 10,000 points.
- Edge Safety: Be careful at the edges of the pyramid — jumping in a direction
  with no cube will cost you a life. For example, when you are on the single top
  cube at the start of a round, only `bottom-right` and `bottom-left` are safe
  initial moves. `top-right` and `top-left` from the top cube will cause you to
  fall off.
"""

FPS_10_PROMPT: str = """
Your plan can contain 1 to 10 actions. Since each action is executed for 0.1
second, your plan is for the next 0.1 to 1 seconds.

You are encouraged to make shorter plans if an enemy is close, as shorter plans
(fewer actions) will get feedback sooner.

An example of response is:
thought: your thought here
move: [bottom-right, bottom-right]
"""

ACTION_MAP = {
    "noop": 0,
    "top-right": 2,
    "bottom-right": 3,
    "top-left": 4,
    "bottom-left": 5,
}

SKIP_SECONDS = 4
