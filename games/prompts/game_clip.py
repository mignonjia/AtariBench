"""GameClip data structures."""

import dataclasses
import json
import textwrap
from typing import Any, Tuple

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

from google3.gdm.agents.eureka.atari_bench.prompt import common_prompt
from google3.gdm.agents.eureka.atari_bench.utils import logging_utils
from google3.gdm.agents.eureka.atari_bench.utils import parse_response


@dataclasses.dataclass
class Step:
  """Represents a single step in a game clip."""

  obs: np.ndarray
  reward: float
  info: dict[str, Any]
  lost_life: bool
  img: PIL.Image.Image = dataclasses.field(init=False)

  def __post_init__(self):
    self.img = logging_utils.obs_to_img(self.obs)


class GameClip:
  """A clip of a game is a sequence of state action within a time range.

  We put a new game clip after each LLM response.

  A game clip contains:
  - start image
  - think process
  - action
  - sequences of states, as well as their rewards, after the action
  These elements are wrapped into their time range. This also means that the
  last frame of the clip is the same as the first frame of the next clip.
  """

  def __init__(
      self,
      start_obs: np.ndarray,
      start_frame_idx: int,
      start_reward: float,
      response_obj: parse_response.LLMResponseList,
      use_full_traj: bool = False,
  ):
    self.start_obs = start_obs
    self.start_img = logging_utils.obs_to_img(start_obs)
    self.start_frame_idx = start_frame_idx
    self.start_reward = start_reward
    self.response_obj = response_obj
    self.action_cnt = len(response_obj.action_ids)
    self.steps_after_action: list[Step] = []
    self.img_w_text_list = []
    self.use_full_traj = use_full_traj
    self.clip_prompt_str = ""

  def update_step(
      self,
      obs: np.ndarray,
      reward: float,
      info: dict[str, Any],
      flag_lost_lives: bool,
  ):
    """Updates the clip with a single step's observation and rewards."""
    self.steps_after_action.append(
        Step(
            obs=obs,
            reward=reward,
            info=info,
            lost_life=flag_lost_lives,
        )
    )

  def _get_state_reward_prompt(self, frame_idx: int, step: Step) -> str:
    """Generates a prompt string for one step's state and reward.

    Args:
      frame_idx: The index of the current frame.
      step: The Step object containing observation, reward, and lost_life info.

    Returns:
      A string containing timestamp, IMG_HOLDER, and feedback.
    """
    # Generate prompt string including timestamp and IMG_HOLDER.
    state_reward_str = common_prompt.STATE_REWARD_TEMPLATE.format(
        TIME=logging_utils.idx_to_time_str(frame_idx),
    )

    # Append feedback based on reward.
    reward = step.reward
    if reward > 0.0:
      state_reward_str += "Feedback: You received a positive score!\n"
    if reward < 0.0:
      state_reward_str += "Feedback: You received a negative score!\n"

    # Append feedback based on lost_life.
    lost_life = step.lost_life
    if lost_life:
      state_reward_str += (
          "Feedback: A new life has begun! This occurs either after you lose a"
          " life or when the current episode ends.\n"
      )
    return state_reward_str

  def _generate_clip_prompt(self):
    """Convert the clip data to a prompt string with a list of imgs.

    This is called by the trajectory class when a new clip is added. In the
    prompt, IMG_HOLDER will be replaced by the img later on.
    """
    # Get states and rewards after each action.
    state_reward_list = []
    for i in range(self.action_cnt):
      frame_idx = self.start_frame_idx + i + 1
      state_reward_str = self._get_state_reward_prompt(
          frame_idx, self.steps_after_action[i]
      )
      state_reward_list.append(state_reward_str)
    list_of_state_reward = "".join(state_reward_list)

    # Construct the prompt string for this clip.
    if self.use_full_traj:  # Append-only multi-turn prompt.
      self.clip_prompt_str = common_prompt.MULTI_TURN_APPENDED_TEMPLATE.format(
          LIST_OF_STATE_REWARD_TEMPLATE=list_of_state_reward
      )
    else:  # Single-turn prompt.
      start_time = logging_utils.idx_to_time_str(self.start_frame_idx)
      end_frame_idx = self.start_frame_idx + self.action_cnt
      end_time = logging_utils.idx_to_time_str(end_frame_idx)
      self.clip_prompt_str = common_prompt.SINGLE_TURN_CLIP_TEMPLATE.format(
          START_TIME=start_time,
          END_TIME=end_time,
          ACTIONS_STR=self.response_obj.action_strings,
          LIST_OF_STATE_REWARD_TEMPLATE=list_of_state_reward,
      )

  def get_clip_prompt(self) -> Tuple[str, list[int]]:
    """Get the clip prompt string and image index list.

    Returns:
      A tuple (clip_prompt_str, img_idx_list), where clip_prompt_str is the
      prompt string for this clip (containing IMG_HOLDER placeholders), and
      img_idx_list contains indices to retrieve images from the trajectory to
      replace the placeholders.
    """
    if not self.clip_prompt_str:
      self._generate_clip_prompt()
    img_idx_list = [
        self.start_frame_idx + i for i in range(1 + self.action_cnt)
    ]
    return self.clip_prompt_str, img_idx_list

  def _get_image_and_reward_for_step(self, step_index: int):
    """Gets the image and reward for a given step index."""
    if step_index == 0:
      return self.start_img, self.start_reward
    else:
      step = self.steps_after_action[step_index - 1]
      return step.img, step.reward

  def _construct_log_text_for_step(self, step_index: int, reward: float) -> str:
    """Constructs the log text for a given step index and reward."""
    if step_index < self.action_cnt:
      wrapped_thought = textwrap.fill(
          f"thought: {self.response_obj.thought_str}", width=80
      )
      return (
          f"reward for this frame: {reward}\n\n"
          f"action_strings: {self.response_obj.action_strings}\n"
          f"current is action {step_index}:"
          f" {self.response_obj.action_strings[step_index]}\n"
          f"{wrapped_thought}"
      )
    else:
      return f"reward for this frame: {reward}\n"

  def _add_text_to_img(
      self, img: PIL.Image.Image, text: str = "default text"
  ) -> PIL.Image.Image:
    """Adds text to the right side of a PIL Image, and return the new image."""
    extra_space_for_text = 500  # The width of the area for the text in pixels.
    padding = 10  # Padding around the text.

    # 1. Create a new image with extra space.
    new_width = img.width + extra_space_for_text
    new_height = img.height
    new_image = PIL.Image.new("RGB", (new_width, new_height), "white")

    # 2. Paste the original image onto the new image.
    new_image.paste(img, (0, 0))

    # 3. Add the text to the new image.
    draw = PIL.ImageDraw.Draw(new_image)
    try:
      font = PIL.ImageFont.truetype("arial.ttf", 12)
    except IOError:
      font = PIL.ImageFont.load_default()

    text_x = img.width + padding
    text_y = padding
    draw.text((text_x, text_y), text, fill="black", font=font)
    return new_image

  def generate_imgs_w_text(self):
    """Generates the list of images with text for the clip."""
    if self.img_w_text_list:
      return
    for i in range(self.action_cnt + 1):
      img, reward = self._get_image_and_reward_for_step(i)
      log_text = self._construct_log_text_for_step(i, reward)
      img_w_text = self._add_text_to_img(img, log_text)
      self.img_w_text_list.append(img_w_text)

  def log_clip_info(self, human_readable: bool = True) -> str:
    """Generates logs for the current game clip in text or JSON-text format."""
    log_data = self.response_obj._asdict()
    log_data.update({
        "rewards": [s.reward for s in self.steps_after_action],
        "lost_lives": [s.lost_life for s in self.steps_after_action],
        "info": [s.info for s in self.steps_after_action],
    })
    if human_readable:
      log_str = "========================================================\n"
      log_str += "\n".join(f"{key}: {value}" for key, value in log_data.items())
      log_str += "\n"
    else:
      log_str = json.dumps(log_data) + "\n"
    return log_str
