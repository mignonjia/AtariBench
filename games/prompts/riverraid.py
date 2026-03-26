"""Riverraid game prompt.

https://ale.farama.org/environments/Riverraid/
"""

GAME_PROMPT: str = """
Riverraid Quick Guide:

Goal: Destroy enemy targets, refuel your jet, and survive as long as possible to
maximize your score.

Available actions:
- noop: Do nothing.
- fire: Fire missiles.
- up: Accelerate the jet.
- right: Move the jet to the right.
- left: Move the jet to the left.
- down: Decelerate the jet.
- upright: Accelerate and move right.
- upleft: Accelerate and move left.
- downright: Decelerate and move right.
- downleft: Decelerate and move left.
- upfire: Accelerate and fire.
- rightfire: Move right and fire.
- leftfire: Move left and fire.
- downfire: Decelerate and fire.
- uprightfire: Accelerate, move right, and fire.
- upleftfire: Accelerate, move left, and fire.
- downrightfire: Decelerate, move right, and fire.
- downleftfire: Decelerate, move left, and fire.

Your Task:

Based on the current state of the game (which will be provided to you), create
your response in the following format:

thought: [Your reasoning about the game state]
move: [action_1, action_2, ..., action_n]

{FPS_SPECIFIC_PROMPT}

Action Planning Guidelines:

- At the very start of the game, the screen will be static. You must perform any
  action (e.g., fire or moving the jet) to begin the mission.
- Your primary objective is to score points by destroying enemy targets: Bridges
  (500), Jets (100), Fuel Depots (80), Helicopters (60), and Tankers (30).
- Survival is critical. Any green area on the screen is a river bank or an
  island; colliding with these green areas will lose a life.
- You will also lose a life if you collide with any enemy unit (except Fuel
  Depots) or if you run out of fuel.
- You have a constantly depleting fuel supply. To refuel, you must fly over a
  Fuel Depot. When your fuel is below 25%, a warning alarm will sound, and
  refueling becomes your highest priority. Also, fuel depot can be destroyed by
  your missiles, so try to avoid shooting into them if you want to refuel.
- Flying slower over a Fuel Depot replenishes more fuel. Use the down action to
  decelerate before flying over a depot.
- Your speed is controlled by the up (accelerate) and down (decelerate) actions.
  Use slower speeds for better control in tight channels and faster speeds to
  advance quickly.
- Your top priority should always be survival. It is always better to avoid a
  collision or secure fuel than to destroy a low-value enemy.
- As the game progresses, fuel depots become less frequent. Prioritize reaching
  the next depot over destroying every enemy on screen.
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

SKIP_SECONDS = 2.0

MAX_LOST_LIVES = 3
