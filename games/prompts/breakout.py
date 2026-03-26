"""Breakout game prompt.

https://ale.farama.org/environments/breakout/
"""

GAME_PROMPT: str = """
BREAKOUT Quick Guide:

Goal: Rewards are earned by hitting bricks with a ball controlled by a paddle
while avoiding the ball falling down. A player has five lives.

Available actions:
- noop: Do nothing. The paddle remains stationary.
- start: At the beginning of the same, generate a ball
- left: Move the paddle to the left 
- right: Move the paddle to the right

Your Task:

Based on the current state of the game (which will be provided to you), decide
on your next actions.

{FPS_SPECIFIC_PROMPT}

Action Planning Guidelines:

- Your objective is to maximize total reward within the fixed game window.

- If no ball (a small red dot) is on the screen, the first action must be
start to serve a new ball into play.

- If there is a ball on the screen, that means the game has already started.
Only press left, right, or noop in this case. The ball is small so pay attention

- The choice between left and right should be based on the ball's current
trajectory and horizontal position. Try to infer the trajectory based on
the historical images.

- Losing a life does not directly reduce score, but it wastes time because the
  ball must be relaunched and scoring opportunities are interrupted. Avoid life
  loss whenever possible.

- For action durations exceeding 0.1 seconds, the left and right movements will
speed up after the first 0.1 seconds, then the speed will remain the same.

- After a left or right command, expect the paddle to continue moving slightly
for an additional 0.1 seconds no matter what next action you choose.
"""

FPS_10_PROMPT: str = """
Your move can contain 1 to 10 actions. Each action is executed for 0.1
second. As a result, your move is for the next 0.1 to 1 seconds.

You are encouraged to plan for less actions if the ball is low and close, as
shorter plans (fewer actions) will get feedback sooner.

An example of response is:
thought: your thought here
move: [right, right, right]
"""

ACTION_MAP = {
    "noop": 0,
    "start": 1,
    "right": 2,
    "left": 3,
}
