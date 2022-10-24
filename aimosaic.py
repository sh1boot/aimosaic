#!/usr/bin/env python3
'''
  A thing.
'''
import argparse
import collections
import datetime
import os
import pathlib
import random
import string

import pydantic
import replicate
from PIL import Image
import imagestuff

from imagestuff import Point, Size

_NOP = False

_SWATCH = Size(512, 512)
_PROMPT_STRENGTH = 0.7
_GUIDANCE_SCALE = 7.5
_STEPS = 25
_NOISE = 0.05

_CANVAS = None

_STAMP = datetime.datetime.now().isoformat('T', 'seconds').replace(':', '')

_PATCH = None

def mkfilename(suffix):
  return pathlib.Path(f'outpaint_{_STAMP}_{suffix}.png')

def add_image(image, position=None):
  global _PATCH
  if _PATCH is None: _PATCH = image
  if position is None:
    ofs_range = Size(_CANVAS.size) - Size(image.size) + Size(1)
    position = Point.xymap(random.randrange, ofs_range)
  # TODO: check overlapping background, and normalise the colours of the
  # incoming image to match what's being replaced.
  # TODO: why can't I pass `position` directly?
  tup = tuple(position)
  _CANVAS.paste(image, tup)

def get_image(position):
  box = (*position, *(position + _SWATCH))
  return _CANVAS.crop(box=box)

def predict(prompt, start=None, alpha=None):
  if start:
    startpath = mkfilename('start')
    start.save(startpath, 'PNG')
  if alpha:
    alphapath = mkfilename('mask')
    alpha.save(alphapath, 'PNG')

  retries = 0
  for try_prompt in [
      prompt,
      prompt,
      f'{" ".join([ f"{word[1:]}-{word[0]}ay" for word in prompt.split(" ") ])}',
      prompt,
      f'{" ".join(sorted(prompt.split(" ")))}',
      f'happy little {prompt} accident',
  ]:
    try:
      model = replicate.models.get("stability-ai/stable-diffusion")
      if start and alpha:
        results = model.predict(
            prompt=try_prompt,
            width=_SWATCH[0],
            height=_SWATCH[1],
            init_image=startpath,
            mask=alphapath,
            prompt_strength=_PROMPT_STRENGTH,
            num_outputs=1,
            num_inference_steps=_STEPS,
            guidance_scale=_GUIDANCE_SCALE,
            seed=None
        )
        os.remove(startpath)
        os.remove(alphapath)
        imagestuff.log(start)
        imagestuff.log(alpha)
      else:
        results = model.predict(
            prompt=try_prompt,
            width=_SWATCH[0],
            height=_SWATCH[1],
            prompt_strength=_PROMPT_STRENGTH,
            num_outputs=1,
            num_inference_steps=_STEPS,
            guidance_scale=_GUIDANCE_SCALE,
            seed=None
        )
      break
    except replicate.exceptions.ModelError as exc:
      if 'NSFW' in str(exc):
        print(f'prompt "{try_prompt[:16]}..{try_prompt[-16:]}" NSFW error, retry: {retries}')
      else:
        raise
    except pydantic.ValidationError:
      print('pydantic is doing that thing again.')

  else:
    raise RuntimeError('retries exceeded trying to escape NSFW loop')
  results = [ imagestuff.download(url) for url in results ]
  for image in results:
    imagestuff.log(image)
  return results


def update_canvas(position, prompt):
  original = get_image(position)
  start = original

  # fully-transparent gives entropy=2.0?
  entropy = start.convert('L').entropy()
  if entropy < 0.05:
    start = None
    alpha = None
  else:
    start = imagestuff.outpaint(start, extra_noise=_NOISE, patch=_PATCH)
    alpha = start.getchannel("A").convert(mode="L")
    start.putalpha(255)
    start = start.convert('RGB')

  if _NOP:
    result = original
  else:
    images = predict(prompt, start, alpha)
    result = images[0]
  add_image(result, position)


def outpaint(start_image, prompt_gen, coord_gen, save_progress=False):
  if start_image: add_image(start_image, next(coord_gen))
  for position in coord_gen:
    imagestuff.reset_log()
    while True:
      prompt = next(prompt_gen)
      print(f'{position} prompt: {prompt}')
      try:
        update_canvas(position, prompt)
        break
      except RuntimeError:
        print('dropping prompt to try again')

    if save_progress:
      imagestuff.save_log(mkfilename(position))
    if not _NOP:
      _CANVAS.save(mkfilename('result'), 'PNG')


def random_coords(area, step, swatch):
  subdiv = ((area - swatch) / step).ceil()
  step = (area - swatch) // subdiv

  results = []
  for x_pos in range(subdiv[0] + 1):
    for y_pos in range(subdiv[1] + 1):
      point = Point(x_pos, y_pos) * step
      results.append(point)
  random.shuffle(results)
  yield from results


def spiral_coords(area, step, swatch):
  subdiv = ((area - swatch) / step).ceil()
  step = (area - swatch) // subdiv

  centre = subdiv // 2
  ncentre = subdiv - centre

  def result(x, y):
    offset = centre + Point(x, y)
    if all(Point(0) <= offset) and all(offset <= subdiv):
      yield offset * step

  yield from result(0, 0)
  for r in range(1, max(ncentre) + 1):
    for x in range(-r + 1, r):
      yield from result(x, -r)
    for y in range(-r, r):
      yield from result(r, y)
    for x in range(r, -r, -1):
      yield from result(x, r)
    for y in range(r, -r, -1):
      yield from result(-r, y)
    yield from result(-r, -r)


def main(opt):
  start_image = None
  if opt.start_image:
    if opt.start_image.startswith('http'):
      start_image = imagestuff.download(opt.start_image)
    else:
      start_image = Image.open(opt.start_image)
  outpaint(
      start_image,
      prompt_gen=PromptGen(opt.prompt, opt.prompt_file),
      #coord_gen=random_coords(opt.canvas, opt.step, opt.swatch),
      coord_gen=spiral_coords(opt.canvas, opt.step, opt.swatch),
      save_progress=(opt.save_progress and not opt.nop)
  )
  if opt.show: _CANVAS.show()


class BackrefGen:
  def __init__(self, source=None, shuffle=False, maxlen=10):
    self.history = collections.deque(maxlen=maxlen)
    self.shuffle = shuffle
    self.iterator = None
    self.src = source

  @classmethod
  def _random_forever(cls, src):
    while True:
      yield random.choice(src)

  @classmethod
  def _linear_forever(cls, src):
    while True:
      yield from src

  def __iadd__(self, other):
    if self.src is None:
      self.src = other
    else:
      self.src.append(other)

  def __iter__(self):
    if self.src is None: raise IndexError
    if self.shuffle:
      src = self.src
      if not isinstance(src, list): src = list(src)
      self.iterator = BackrefGen._random_forever(src)
    else:
      self.iterator = BackrefGen._linear_forever(self.src)
    return self

  def __next__(self):
    if self.iterator is None: self.__iter__()
    result = next(self.iterator)
    self.history.appendleft(result)
    return result

  def __getitem__(self, i):
    if not 0 <= i < self.history.maxlen: raise IndexError
    while len(self.history) <= i:
      next(self)
    return self.history[i]

class PromptGen:
  _DEFAULTS = {
      'noun': [ 'kitten', 'puppy', 'goat', 'bicycle' ],
      'verb': [ 'eat', 'climb', 'paint', 'play' ],
      'verbs': [ 'eats', 'climbs', 'paints', 'plays' ],
      'verbing': [ 'eating', 'climbing', 'painting', 'playing' ],
      'adverb': [ 'slowly', 'dangerously', 'skillfully', 'abstractly' ],
      'adjective': [ 'funny', 'fluffy', 'green', 'monstrous' ],
  }
  _DEFAULT_PROMPT = string.Template('$adjective $_noun $verbs $adverb with $noun')

  @classmethod
  def prompt_arg(cls, arg):
    try:
      key, path = arg.split('=', 1)
    except ValueError:
      key = ''
      path = arg
    value = argparse.FileType('r', encoding='utf-8')(path)
    return key, value

  def __init__(self, prompts, sources, prompt_shuffle=True):
    class MyDict(dict):
      def __getitem__(self, key):
        base = key.lstrip('_')
        gen = super().get(base, None)
        if not isinstance(gen, BackrefGen): return super().get(key)

        depth = len(key) - len(base)
        if depth > 0: return gen[depth - 1]
        return next(gen)

    self.sources = MyDict()

    if sources:
      mypunctuation = string.punctuation.replace('_', '')
      for identifier, value in sources:
        key = identifier.rstrip(mypunctuation)
        flags = identifier[len(key):]
        shuffle = '~' in flags
        words = '^' in flags
        if not key: key = 'word' if words else 'line'
        dst = self.sources.setdefault(key, BackrefGen(shuffle=shuffle))
        if words:
          dst += value.read().split()
        else:
          dst += map(str.strip, value.readlines())

    if not prompts:
      if self.sources:
        prompts = [ string.Template(f'${src}') for src in self.sources ]
      else:
        prompts = [ PromptGen._DEFAULT_PROMPT ]
    self.prompts = BackrefGen(prompts, shuffle=prompt_shuffle, maxlen=1)

    for key, value in PromptGen._DEFAULTS.items():
      if key not in self.sources:
        self.sources[key] = BackrefGen(value, shuffle=True)

  def __iter__(self):
    return self

  def __next__(self):
    return next(self.prompts).substitute(self.sources)


def xy_pair(arg):
  try:
    x, y = arg.split(',')
    return Point(int(x), int(y))
  except ValueError as exc:
    raise argparse.ArgumentTypeError(f'not an int pair: "{arg}"') from exc

def wh_pair(arg):
  try:
    x, y = arg.split('x')
    return Size(int(x), int(y))
  except ValueError:
    pass
  return Size(xy_pair(arg))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--canvas', type=wh_pair, default=Size(800,800))
  parser.add_argument('--step', type=wh_pair, default=Size(288,288))
  parser.add_argument('--swatch', type=wh_pair, default=_SWATCH)
  parser.add_argument('--start_image', type=str, default=None)
  parser.add_argument('--prompt', nargs='*', type=string.Template, default=None)
  parser.add_argument('--prompt_file', nargs='*',
                      type=PromptGen.prompt_arg, default=None)

  parser.add_argument('--prompt_strength', type=float, default=_PROMPT_STRENGTH)
  parser.add_argument('--guidance_scale', type=float, default=_GUIDANCE_SCALE)
  parser.add_argument('--steps', type=int, default=_STEPS)

  parser.add_argument('--noise', type=float, default=_NOISE)
  parser.add_argument('--nop', action='store_true', default=_NOP)
  parser.add_argument('--show', action='store_true', default=False)
  parser.add_argument('--save_progress', action='store_true', default=False)
  _opt = parser.parse_args()

  # TODO: no globals
  _CANVAS = Image.new("RGBA", _opt.canvas)
  _SWATCH = _opt.swatch
  _PROMPT_STRENGTH = _opt.prompt_strength
  _GUIDANCE_SCALE = _opt.guidance_scale
  _STEPS = _opt.steps
  _NOP = _opt.nop
  _NOISE = _opt.noise

  main(_opt)
