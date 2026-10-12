"""NameThisGame game prompt.

https://ale.farama.org/environments/name_this_game/
"""

GAME_PROMPT: str = """
Name This Game Quick Guide:

Goal: Shoot the shark and the octopus's tentacles and get as many points as
possible. You also receive points for replenishing your oxygen if the tank is
already full. You should maximize score within the fixed game window.

Available actions (moving and firing):
- noop: Do nothing.
- fire: fire upwards to attack the shark or the octopus's tentacles.
- right: move right
- left: move left
- rightfire: move right and fire
- leftfire: move left and fire

Your Task:

Based on the current state of the game (which will be provided to you), create
your response in the following format:

thought: [Your reasoning about the game state]
move: [action_1, action_2, ..., action_n]

{FPS_SPECIFIC_PROMPT}

Action Planning Guidelines:
- Core Actions:
  - Movement: Move the diver left and right along the sea floor.
  - Firing: Press the fire button to shoot upwards at the shark or the octopus
    tentacles.
  - Replenish Oxygen: Move your diver directly under the white oxygen line
    dropped from the boat above to refill your air supply (the red bar at the
    bottom of the screen).
- Threats & How to Handle Them:
  - The Shark: A shark will constantly try to attack you. Shooting the shark
    will not kill it but will cause it to retreat to the surface before starting
    its attack again.
  - The Octopus: An octopus will lower its tentacles from above. You must shoot
    the lowest segment of a descending tentacle to make it retract.
- Risk Conditions:
  - Running out of oxygen wastes time and prevents scoring.
  - Being touched by the shark disrupts control and costs time.
  - Letting an octopus tentacle reach the sea floor gives up a dangerous lane
    and reduces scoring opportunities.
- Scoring Details:
  - Hitting a tentacle: 1 point
  - Hitting the shark: 1 point
  - Bonus Oxygen: For each unit of air you take in after your tank is already
    full, you receive 1 point.
- Ignore the score at the bottom center of the screen, as I will use the above
  scoring details to calculate the score instead of the original game score.
"""

FPS_10_PROMPT: str = """
Your plan can contain 1 to 10 actions. Since each action is executed for 0.1
second, your plan is for the next 0.1 to 1.0 seconds.

You are encouraged to make shorter plans the shark, oxygen line, or octopus's
tentacles are close, as shorter plans (fewer actions) will get feedback sooner.

An example of response is:
thought: your thought here
move: [right, right, right]
"""

ACTION_MAP = {
    "noop": 0,
    "fire": 1,
    "right": 2,
    "left": 3,
    "rightfire": 4,
    "leftfire": 5,
}

SKIP_SECONDS = 15.0
MAX_FRAMES = 500
