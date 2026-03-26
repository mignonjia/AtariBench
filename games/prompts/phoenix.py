"""Phoenix game prompt.

https://ale.farama.org/environments/phoenix/
"""

GAME_PROMPT: str = """
Phoenix Quick Guide:

Goal: Your goal is to reach and shoot the alien pilot. On your way there, you
must eliminate waves of war birds while avoiding their bombs.

Available actions:
- noop: Do nothing.
- fire: Fire up.
- right: Move right.
- left: Move left.
- down: Open shield.
- rightfire: Move right and fire up.
- leftfire: Move left and fire up.
- downfire: Open shield and fire up.

Your Task:

Based on the current state of the game (which will be provided to you), create
your response in the following format:

thought: [Your reasoning about the game state]
move: [action_1, action_2, ..., action_n]

{FPS_SPECIFIC_PROMPT}

Action Planning Guidelines:

- You control a laser cannon which can fire up, move left or right, or open a
  shield to protect itself. Your goal is to survive the bombing attachs of four
  separate flocks of Phoenix war birds, then shoot the alien pilot in the 
  upcoming alien spaceship.

- To eliminate the Phoenix war birds, you need to fire upwards and hit them. The
  laser bomb you fire will move vertically upwards, so you should fire when the
  target birds are (moving towards a position that is) directly above you. To
  eliminate the large, menacing Phoenixes of the third and fourth waves, you
  have to hit them in the center.  If you only wing a Phoenix bird, it will soon
  regenerate the missing part and become a whole bird to attack you some more.

- Correspondingly, these birds will also drop vertically falling bombs. At this
  time, you should move horizontally to avoid them, or open the shield using the
  "down" action. Regarding the shield, it is worth noting that it can only last
  for 1.5 seconds, and you cannot move left or right, but only shoot upwards
  while it is active. Therefore, if you want to use the shield to defend against
  bombs, you should only open it when the bomb is close. Additionally, the
  shield has a 3.5-second cooldown. Thus, you cannot keep the shield open all
  the time. So, Use your force field sparingly and as a strategic device in
  emergency situations. If a bird brushes up against your force field, it will
  be disintegrated; but if a bird crashes into your unprotected laser cannon,
  you're both "dead ducks."

- Important: Your actions should not use firing commands all the time. Firing
  commands include fire, leftfire, rightfire and downfire. Instead, you should
  alternate between firing commands and non-firing commands. This is because if
  your command always include firing, this is considered continuous firing, and
  continuous firing doesn't work in this game. So at least have one non-firing
  command between two firing commands.

- You will receive score each time you hit a bird, or you destroy the alien
  pilot. Point values are identical for the first two waves of attack by the
  small Phoenix birds. (Figures 1 and 2).  You get 20 points for each small bird
  you hit while it is moving in a horizontal pattern; you get 80 points for each
  bird you shoot while it is actually swooping in at you. 
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
    "right": 2,
    "left": 3,
    "down": 4,
    "rightfire": 5,
    "leftfire": 6,
    "downfire": 7,
}

MAX_LOST_LIVES = 2
