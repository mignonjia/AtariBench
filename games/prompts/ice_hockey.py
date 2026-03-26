"""IceHockey game prompt.

https://ale.farama.org/environments/ice_hockey/
"""

GAME_PROMPT: str = """
IceHockey Quick Guide:

Goal: You control the blue player playing against the yellow player. Your goal
is to score as many points as possible in a standard game of Ice Hockey. The
ball is usually called “the puck”. There are 32 shot angles ranging from the
extreme left to the extreme right. The angles can only aim towards the
opponent’s goal. Just as in real hockey, you can pass the puck by shooting it
off the sides of the rink. This can be really key when you’re in position to
score a goal.

Available actions:
- noop: Do nothing.
- shoot: shoot the ball with your stick.
- up: move up
- right: move right
- left: move left
- down: move down
- upright: move up and right
- upleft: move up and left
- downright: move down and right
- downleft: move down and left
- upshoot: move up and shoot
- rightshoot: move right and shoot
- leftshoot: move left and shoot
- downshoot: move down and shoot
- uprightshoot: move up and right and shoot
- upleftshoot: move up and left and shoot
- downrightshoot: move down and right and shoot
- downleftshoot: move down and left and shoot

Your Task:

Based on the current state of the game (which will be provided to you), create
your response in the following format:

thought: [Your reasoning about the game state]
move: [action_1, action_2, ..., action_n]

{FPS_SPECIFIC_PROMPT}

Action Planning Guidelines:
- Important: You control the two blue players and defend against the two yellow
  players. Your net is at the top of the court.
- The punk is a black dot on the court.
- There are two blue players, and you control the player closest to the puck.
  Roughly speaking, when the puck is at the top of the court, you control the
  upper blue player to defend your net from the yellow player. When the puck is
  at the bottom of the court, you control the lower blue player to attack the
  yellow player's net.
- Offense (Attacking the yellow player's goal - the lower goal):
  - Gain control of the puck by moving near it.
  - Shoot the puck to score in the goal. It will be easier to score if the
    yellow player is not too close to their goal.
  - When shooting, the angle of your shot is determined by the puck's position
    on your stick as it moves back and forth.
    - A shot from the far left of your stick will go to the extreme right of the
      opponent's goal.
    - A shot from the far right of your stick will go to the extreme left of the
      opponent's goal.
  - Be quick from the face-off to gain initial control of the puck.
  - Utilize passing to your teammate to move the puck up the ice effectively.
  - A tip from the game's designer: when you first gain control of a loose puck,
    it's placed on the inside corner of your stick, allowing for an immediate,
    extremely angled shot to surprise your opponent.
- Defense (Protecting your goal - the upper goal):
  - The objective is to prevent the yellow player from shooting the puck into
    your goal.
  - Use your stick to knock the puck away from your opponent.
  - You can also body-check your opponent to slow them down.
  - A more aggressive defensive move is to swing your stick to knock your
    opponent down, giving you a temporary player advantage.
  - Goalies cannot be knocked down in front of their goal.
  - When defending your goal, position your player to cut down the opponent's
    shooting angle.
  - Avoid bringing your goalie out too far from the goal, as a smart opponent
    can score an easy goal by banking a shot off the boards.
"""

FPS_10_PROMPT: str = """
Your plan can contain 1 to 10 actions. Since each action is executed for 0.1
second, your plan is for the next 0.1 to 1.0 seconds.

You are encouraged to make shorter plans when the puck is close to either goal.
Shorter plans (fewer actions) allow you to get feedback from the game state
sooner, enabling you to react more quickly to the fast-paced action.

An example of response is:
thought: your thought here
move: [right, right, right]
"""

ACTION_MAP = {
    "noop": 0,
    "shoot": 1,
    "up": 2,
    "right": 3,
    "left": 4,
    "down": 5,
    "upright": 6,
    "upleft": 7,
    "downright": 8,
    "downleft": 9,
    "upshoot": 10,
    "rightshoot": 11,
    "leftshoot": 12,
    "downshoot": 13,
    "uprightshoot": 14,
    "upleftshoot": 15,
    "downrightshoot": 16,
    "downleftshoot": 17,
}
