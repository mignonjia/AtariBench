"""Robotank game prompt.

https://ale.farama.org/environments/robotank/
"""

GAME_PROMPT: str = """
Robotank Quick Guide:

Goal: Command an advanced Robotank to destroy enemy tanks while avoiding being
hit.

Available actions:
- noop: Do nothing.
- fire: Fire the cannon.
- up: Move the tank forward.
- right: Rotate the tank clockwise (turn right).
- left: Rotate the tank counter-clockwise (turn left).
- down: Move the tank backward.
- upright: Move forward while rotating clockwise.
- upleft: Move forward while rotating counter-clockwise.
- downright: Move backward while rotating clockwise.
- downleft: Move backward while rotating counter-clockwise.
- upfire: Move forward and fire.
- rightfire: Rotate clockwise and fire.
- leftfire: Rotate counter-clockwise and fire.
- downfire: Move backward and fire.
- uprightfire: Move forward, rotate clockwise, and fire.
- upleftfire: Move forward, rotate counter-clockwise, and fire.
- downrightfire: Move backward, rotate clockwise, and fire.
- downleftfire: Move backward, rotate counter-clockwise, and fire.

Your Task:

Based on the current state of the game (which will be provided to you), create
your response in the following format:

thought: [Your reasoning about the game state]
move: [action_1, action_2, ..., action_n]

{FPS_SPECIFIC_PROMPT}

Action Planning Guidelines:

- Your mission is to destroy enemy tanks while preventing them from overrunning
  the city at the bottom of the screen, since that wastes valuable scoring time.
- You command a squadron of 1 Robotanks (lives). Your tank can sustain several
  hits before being destroyed. When your tank is hit, the screen will flash red.
  Losing a life does not directly reduce score, but the recovery costs time and
  momentum. Avoid being hit whenever possible.
- The scanner at the bottom of the screen is your most critical tool. Think of
  it as a top-down radar. Your tank is located at the center of the bottom edge,
  and the white dots represent the positions of enemy tanks on the battlefield
  in front of you.
- Your tank's movement is rotation-based. The `left` and `right` actions cause
  your tank to rotate continuously. This is not an instant turn. You will need
  to apply a sequence of `left` or `right` actions to aim your tank at a target.
  Once you are facing the desired direction, use `up` and `down` to move.
- Use the radar to gauge distance. If enemy tanks are far away, don't just
  rotate in place. Proactively move `up` (forward) to close the distance and
  engage, or move `down` (backward) to reposition. Staying mobile is key to
  controlling the battlefield.
- The battle progresses through a day/night cycle (Day -> Dusk -> Night). During
  the Day, you have full visibility. At Night, your vision is restricted to a
  small searchlight, making the scanner your only way to find enemies.
- Scoring is simple: you earn 1 point for every enemy tank you destroy.
"""

FPS_10_PROMPT: str = """
Your plan can contain 1 to 10 actions. Since each action is executed for 0.1
second, your plan is for the next 0.1 to 1 seconds.

You are encouraged to make shorter plans, as shorter plans (fewer actions) will
get feedback sooner. This can give you more recent game states so that you can
take better actions.

An example of response is:
thought: your thought here
move: [left, left, upfire]
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
