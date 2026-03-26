"""Gopher game prompt.

https://ale.farama.org/environments/gopher/
"""

GAME_PROMPT: str = """
Gopher Quick Guide:

Goal: You control a shovel-wielding farmer who protects a crop of three carrots
from a gopher.

Available actions:
- noop: Do nothing.
- bonk: Strike downwards.
- right: Move right.
- left: Move left.
- rightbonk: Move right and strike downwards.
- leftbonk: Move left and strike downwards.

Your Task:

Based on the current state of the game (which will be provided to you), create
your response in the following format:

thought: [Your reasoning about the game state]
move: [action_1, action_2, ..., action_n]

{FPS_SPECIFIC_PROMPT}

Action Planning Guidelines:

- You control a shovel-wielding farmer who protects a crop of three carrots from
  a gopher. The Gopher is located underground, while the farmer and carrots are
  on the surface. The Farmer can only move horizontally. The Gopher will dig
  tunnels upwards. When a tunnel reaches the surface, the Gopher can emerge from
  the hole and run towards the carrots. When it touches a carrot, you lose that
  carrot. Losing a carrot does not directly reduce score, but it damages your
  defensive position and wastes time recovering control.

- As the Farmer, you need to defend against the Gopher. You can: 1) fill the 
  holes; 2) bonk the Gopher.

- You can only fill a hole when the hole is up to the surface. To fill a hole,
  you must first position the farmer directly above the hole and then press
  'bonk' to fill it. A hole that reaches the surface requires multiple 'bonk'
  actions to be completely filled, but even a partially filled hole provides
  some defensive benefit. You can check the images to see how much the hole is
  filled.

- You can Bonk the Gopher on the head as he emerges from his hole or anywhere
  above ground as he runs for a carrot. To bonk the Gopher, you must wait until
  its head is above ground, or when it is rushing towards a carrot. You need to
  be positioned directly above it and 'bonk'. If you choose to shot it when it
  is running towards a carrot, just pay attention that the Gopher moves very
  quickly, so you need be careful and shot it when it reach under your shovel.
  Note that there is a cooldown every time you bonk, so you must be careful to
  bonk when needed, otherwise you might experiencing a cooldown time and can't
  bonk in time.

- You will receive score each time you fill a hole or bonk the gopher. The 
  second action, in particular, will give you a very high reward.
"""

FPS_10_PROMPT: str = """
Your plan can contain 1 to 10 actions. Since each action is executed for 0.1
second, your plan is for the next 0.1 to 1 seconds.

You are encouraged to make shorter plans, as shorter plans (fewer actions) will
get feedback sooner. This can give you more recent game states so that you can
take better actions.

An example of response is:
thought: your thought here
move: [right, right, bonk]
"""

ACTION_MAP = {
    "noop": 0,
    "bonk": 1,
    "right": 3,
    "left": 4,
    "rightbonk": 6,
    "leftbonk": 7,
}

SKIP_SECONDS = 8.0
