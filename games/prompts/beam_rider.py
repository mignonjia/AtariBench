"""BeamRider game prompt.

https://ale.farama.org/environments/beam_rider/
"""

GAME_PROMPT: str = """
BeamRider Quick Guide:

Goal: You control a space-ship that travels forward at a constant speed. You can
only steer it sideways between discrete positions. Your goal is to destroy enemy
ships, avoid their attacks and dodge space debris.

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
- Your goal is to destroy enemy ships, avoid their attacks and dodge space
  debris. The number at the top center of the screen is your score; the higher,
  the better.
- Among the objects approaching from the front, enemy ships are white and can be
  destroyed by your fire. You must dodge all other objects (including enemy
  attacks, space debris, etc.). Since these objects only move along fixed
  tracks, you just need to move horizontally to a different discrete position to
  evade them.
- To attack enemy ships, you must align with their track and then fire.
- If the game has started but there are no enemies, you can move left or right
  to officially start the game.
- The yellow spaceships in the bottom-left corner indicate your remaining lives.
  Although there are multiple lives as shown in the image, we will end the game
  after the first loss (i.e. when you are hit by an enemy ship).
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
    "right": 3,
    "left": 4,
    "rightfire": 7,
    "leftfire": 8,
}

SKIP_SECONDS = 8.0

# gemini-2.5-flash: 1 life = ~800 frames
MAX_LOST_LIVES = 1
