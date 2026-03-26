"""FishingDerby game prompt.

https://ale.farama.org/environments/fishing_derby/
"""

GAME_PROMPT: str = """
Sequest Quick Guide:

Goal: Be the first fisherman to catch fish while avoiding the hungry shark.

Available actions:
- noop: Do nothing.
- fire: Reel in the fish faster.
- up: Move the fishing line upwards.
- right: Move the fish line towards right.
- left: Move the fishing line towards left.
- down: Move the fishing line downwards.
- upright: Move the fishing line diagonally up-right.
- upleft: Move the fishing line diagonally up-left.
- downright: Move the fishing line diagonally down-right.
- downleft: Move the fishing line diagonally down-left.
- upfire: Move the fishing line upwards and reel in.
- rightfire: Move the fishing line towards right and reel in.
- leftfire: Move the fishing line towards left and reel in.
- downfire: Move the fishing line downwards and reel in.
- uprightfire: Move the fishing line diagonally up-right and reel in.
- upleftfire: Move the fishing line diagonally up-left and reel in.
- downrightfire: Move the fishing line diagonally down-right and reel in.
- downleftfire: Move the fishing line diagonally down-left and reel in.

Your Task:

Based on the current state of the game (which will be provided to you), create
your response in the following format:

thought: [Your reasoning about the game state]
move: [action_1, action_2, ..., action_n]

{FPS_SPECIFIC_PROMPT}

Action Planning Guidelines:

- Your Role: You control the fisherman on the left pier.
- How to Fish: You control a fishing line. Move it left, right, up, and down to
  position the hook directly in front of a fish's mouth to get a bite. You can't
  catch it by simply standing still in front of the fish. You need to move the
  line to catch the fish.
- Movement Boundaries: Your operational area is limited to the left half of the
  screen. You cannot move your fishing line past the center to catch fish on the
  opponent's side.
- After hooking: When both players have hooked a fish, only one can reel up at a
  time (the first one hooked). The other fish will swim up slowly until the
  first fish has either been caught or eaten by the shark.
- Scoring: There are six rows of fish. The top two rows are worth 2 pounds each,
  the middle two are worth 4 pounds each, and the most valuable bottom two rows
  are worth 6 pounds each.
- Object Identification: Note the yellow text 'ACTIVISION' at the bottom left of
  the screen. This is a brand logo and not a fish; do not attempt to catch it.
- Defining a 'Hooked' Fish: A fish is successfully "hooked" when it attaches to
  your fishing line. Visually, you will see the fish stop its horizontal
  swimming pattern and begin to be pulled vertically upwards by your line.
- Correct Use of the 'fire' Action: Crucially, only use the fire action (or any
  action combined with fire, like upfire, leftfire, etc.) after a fish is
  hooked, as defined above. The fire action's purpose is to reel in a hooked
  fish faster. Using fire when no fish is hooked is an ineffective action.
- Watch Out for the Shark: While reeling in a hooked fish, a shark may appear.
  Maneuver your line left and right (e.g., using leftfire or rightfire) to evade
  the shark as you reel in your catch. You can also control the speed of
  reeling. If the shark eats your fish, you get no points.
- Strategy: The bigger, higher-value fish are in the deeper water. Aim for these
  to score points faster, but be mindful that the longer reeling time gives the
  shark more opportunity to attack. If the bottom rows on your side are empty,
  immediately shift your focus to catching fish from the upper, less valuable
  rows. It is better to score some points than to wait and score none.
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

SKIP_SECONDS = 0

MAX_FRAMES = 500
