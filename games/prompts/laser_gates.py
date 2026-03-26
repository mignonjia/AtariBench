"""LaserGates game prompt.

https://ale.farama.org/environments/laser_gates/
"""

GAME_PROMPT: str = """
Laser Gates Quick Guide:

Goal: Navigate the Dante Dart ship through the caverns. Your goal is to avoid
collisions with the cavern walls, enemies and enemy fire to avoid losing
shields. You will end the game if your shield is depleted. You can earn scores
by destroying the enemies and successfully passing through Forcefields.

Available actions (moving and firing):
- noop: Do nothing.
- fire: fire horizontally.
- up: move up
- right: move right
- left: move left
- down: move down
- upright: move up and right
- upleft: move up and left
- downright: move down and right
- downleft: move down and left
- upfire: move up and fire
- rightfire: move right and fire
- leftfire: move left and fire
- downfire: move down and fire
- uprightfire: move up and right and fire
- upleftfire: move up and left and fire
- downrightfire: move down and right and fire
- downleftfire: move down and left and fire

Your Task:

Based on the current state of the game (which will be provided to you), create
your response in the following format:

thought: [Your reasoning about the game state]
move: [action_1, action_2, ..., action_n]

{FPS_SPECIFIC_PROMPT}

Action Planning Guidelines:
- Core Actions:
  - Movement: Pilot the ship in any direction to avoid collisions with walls and
    enemies. The ship can be faced left or right.
  - Firing: Press the fire button to shoot lasers in the direction the ship is
    facing. This is used to destroy enemies and certain obstacles.

- Resource Management:
  - Shields (Health): You begin with 24 shield units. Collisions and being hit
    by enemy fire will reduce shields.
    If the shield indicator flashes red, the next hit is fatal.
    - Shield Loss:
      - Computer Wall: 1 Unit
      - Shot from Rock Muncher or Radar Mortar: 1 Unit
      - Byte Bat, Rock Muncher, Homing Missile, Radar Cannon, Densepack Column,
        any Forcefield or Detonator: 6 Units
    - When the Shield Indicator flashes red, another collision will destroy the
      Dante Dart.
    - You can check current shield count by looking at the shield indicator in
      the image.
  - Energy (Fuel): Energy is consumed constantly. You must touch the "Energy
    Pods" that appear to replenish it. Do NOT shoot the Energy Pods.

- Threats & Obstacles:
  - Enemies (Destroy): Shoot enemies like Radar Mortars, Rock Munchers, Homing
    Missiles, and Byte Bats. Shoot the enemy to earn points.
  - Forcefields (Avoid/Time): Carefully time your movement to fly through the
    gaps in the three types of forcefields (Flashing, Flexing, Fixed). Do not
    touch the forcefields themselves.
  - Densepack Columns (Destroy): These grey columns block your path and must be
    destroyed with laser fire.
  - Cavern Walls (Avoid): Colliding with the walls will deplete your shields.

- Destroying Detonators:
  - The primary target is a large grey object with "6507" on it. To destroy it,
    you must shoot the pins on its side.
  - Critical Rule: Some pins are booby-trapped. Hitting a booby-trapped pin
    depletes shields. Hitting the same booby-trapped pin twice will instantly
    destroy your ship. The location of these traps is random in each new game.

- Ignore the score at the bottom center of the screen, as I will use the above
  scoring details to calculate the score instead of the original game score.

- Positive Scores
  - Pass Forcefield
  - Destroy Radar Mortar
  - Destroy Rock Muncher
  - Destroy Bat
  - Destroy Homing Missile
  - Destroy Detonator
"""

FPS_10_PROMPT: str = """
Your plan can contain 1 to 10 actions. Since each action is executed for 0.1
second, your plan is for the next 0.1 to 1.0 seconds.

You are encouraged to make shorter plans when the laser beam or enemy or
fireball is close, as shorter plans (fewer actions) will get feedback sooner.

An example of response is:
thought: your thought here
move: [right, right, right]
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
