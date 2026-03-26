"""Configuration for common prompt templates in ALE environments.

The prompt is chosen between
  (1). SINGLE_TURN_PROMPT_TEMPLATE
  (2). MULTI_TURN_PROMPT_TEMPLATE + MULTI_TURN_APPENDED_TEMPLATE based on
  args.use_full_traj.

- For SINGLE_TURN_PROMPT_TEMPLATE, the historical context includes the non-zero
  reward history and recent history.
    - LIST_OF_REWARD_CLIPS_TEMPLATE will be replaced by REWARD_CLIPS_TEMPLATE
      if non-zero reward history is available. Otherwise, it will be replaced by
      an empty string.
    - LIST_OF_CLIP_TEMPLATE will be replaced by a list of CLIP_TEMPLATE.

- For MULTI_TURN_PROMPT_TEMPLATE, we just use the current game state. Then after
  each turn, we append the new states and rewards for the actions taken using
  MULTI_TURN_APPENDED_TEMPLATE, and ask the model to make the next actions.

- GAME_PROMPT are defined in the game specific prompt files.

- GAME_OVER_PROMPT are defined in termination.py.
"""

SINGLE_TURN_PROMPT_TEMPLATE: str = """
You are an intelligent AI player playing a game. Your goal is to make progress
and maximize total reward within the fixed time budget.

{GAME_PROMPT}
{GAME_OVER_PROMPT}

To make your decision, you will be provided with both historical context from
your current game session and the current, up-to-the-moment game state.

** Historical Context **

The historical data is organized into units called "clips." A clip represents
the complete sequence of events from one of your previous move responses and
contains:

* Actions: The list of actions you chose.
* Game States: The sequence of game images with timestamps that resulted from
  your actions.
* Feedback: This includes positive or negative scores, or notification that a
  life was lost and play restarted. If nothing significant happened or there
  was no score change, no feedback will be attached.

You will receive two types of history built from clips:

* Non-Zero Reward History (if available): Highlights from your current gameplay
  showing the specific moments where your actions led directly to non-zero
  rewards. This information helps you understand the game's reward mechanics
  and how to maximize your chances of earning more.
* Recent History: The most recent clips of gameplay, showing your last actions
  and the immediate outcomes. This provides context for the current game
  trajectory.

** Current Game State **

* Current State: A single, up-to-the-moment snapshot of the game, including the
  current timestamp and screen image. This is the state from which you must
  plan your immediate next actions.

Below is the non-zero reward history (if available) to help understand the
game's reward mechanics:
{LIST_OF_REWARD_CLIPS_TEMPLATE}

Below is the recent history to help understand the recent game trajectory:
<recent_history>
{LIST_OF_CLIPS_TEMPLATE}
</recent_history>

Below is the current state:
time: {CURRENT_TIME}
IMG_HOLDER

Now please predict the next actions.

IMPORTANT: You MUST format your response using EXACTLY these lines:

thought: [Your reasoning about the game state]
move: [action_1, action_2, ..., action_n]
"""


MULTI_TURN_PROMPT_TEMPLATE: str = """
You are an intelligent AI player playing a game. Your goal is to make progress
and maximize total reward within the fixed time budget.

{GAME_PROMPT}
{GAME_OVER_PROMPT}

To make your decision, you will be provided with both historical context from
your current game session and the current, up-to-the-moment game state.

After each turn, you will receive the new states and rewards for the actions
you took, and you will be able to choose your next actions based on these new
information.

Below is the initial state:
time: {CURRENT_TIME}
IMG_HOLDER

Now please predict the next actions.

IMPORTANT: You MUST format your response using EXACTLY these lines:

thought: [Your reasoning about the game state]
move: [action_1, action_2, ..., action_n]
"""

MULTI_TURN_APPENDED_TEMPLATE: str = """
states and feedbacks after your actions:

{LIST_OF_STATE_REWARD_TEMPLATE}

Now please choose your actions following the same format in the initial prompt.
"""

REWARD_CLIPS_TEMPLATE: str = """
<non_zero_reward_history>
{LIST_OF_CLIPS_TEMPLATE}
</non_zero_reward_history>
"""

SINGLE_TURN_CLIP_TEMPLATE: str = """
<clip start="{START_TIME}" to end="{END_TIME}">
<start_state>
IMG_HOLDER
</start_state>
<actions>{ACTIONS_STR}</actions>
<states_and_rewards_after_actions>
{LIST_OF_STATE_REWARD_TEMPLATE}
</states_and_rewards_after_actions>
</clip>
"""

STATE_REWARD_TEMPLATE: str = """
time: {TIME}
IMG_HOLDER
"""
