"""Tennis game prompt.

https://ale.farama.org/environments/tennis/
"""

GAME_PROMPT: str = """
Tennis Quick Guide:

Goal: You control the orange player playing against the blue player.

Available actions:
- noop: Do nothing.
- serve: Serve the ball when a round start. After serving, players will swing
  automatically if they are in position to return a shot. So only press serve
  when a round starts.
- left: move left
- right: move right
- up: move up
- down: move down
- upright: move up and right
- upleft: move up and left
- downright: move down and right
- downleft: move down and left

Your Task:

Based on the current state of the game (which will be provided to you), create
your response in the following format:

thought: [Your reasoning about the game state]
move: [action_1, action_2, ..., action_n]

{FPS_SPECIFIC_PROMPT}

Action Planning Guidelines:
- Important: You controls the orange player, and your position is the top half
  of the court.
- Make sure to press serve when a round starts and you are the one to serve.
- After the blue player hits the ball, you can watch the ball's path and decide
  which direction to move to catch the ball.
- You don't need to press serve again to catch the ball after the first serve.
  You will swing automatically if you are in position to return a shot.
- If the ball hit the net, you will lose this round. A good strategy is to
  swing far away from the net.
"""

FPS_10_PROMPT: str = """
Your plan can contain 1 to 10 actions. Since each action is executed for 0.1
second, your plan is for the next 0.1 to 1.0 seconds.

You are encouraged to make shorter plans after the blue player hits the ball,
so you can closely monitor the states and make better next steps.

An example of response is:
thought: your thought here
move: [right, right, right]
"""

ACTION_MAP = {
    "noop": 0,
    "serve": 1,
    "up": 2,
    "right": 3,
    "left": 4,
    "down": 5,
    "upright": 6,
    "upleft": 7,
    "downright": 8,
    "downleft": 9,
}

MAX_LOST_REWARDS = 3
