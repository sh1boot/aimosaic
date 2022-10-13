#!/usr/bin/env python3
import argparse
import datetime
import os
import pathlib
import pydantic
import random
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
  global _PATCH
  original = get_image(position)
  start = original

  # fully-transparent gives entropy=2.0?
  entropy = start.convert('L').entropy()
  if entropy < 0.05:
    print(f'dropping start images for lack of entropy {entropy}.')
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
  if _PATCH is None: _PATCH = result
  add_image(result, position)


def outpaint(start_image, prompt_gen, coord_gen,
             save_progress=False, prefix=None, suffix=None):
  if start_image: add_image(start_image, next(coord_gen))
  for position in coord_gen:
    imagestuff.reset_log()
    while True:
      varying_prompt = next(prompt_gen)
      prompt = " ".join([p for p in [prefix, varying_prompt, suffix] if p])
      print(f'{position} prompt: {prompt}')
      try:
        update_canvas(position, prompt)
        break
      except RuntimeError:
        print('dropping prompt to try again')

    if save_progress:
      imagestuff.save_log(mkfilename(position))
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


def slow_iter(source, rate):
  for output in iter(source):
    yield output
    while random.randrange(rate) > 0:
      yield output


def main(opt):
  prompts = opt.prompts.readlines()
  prompts = [ prompt.strip() for prompt in prompts ]
  random.shuffle(prompts)
  start_image = None
  if opt.start_image:
    if opt.start_image.startswith('http'):
      start_image = imagestuff.download(opt.start_image)
    else:
      start_image = Image.open(opt.start_image)
  outpaint(
      start_image,
      prompt_gen=slow_iter(prompts, opt.prompt_change_rate),
      #coord_gen=random_coords(opt.canvas, opt.step, opt.swatch),
      coord_gen=spiral_coords(opt.canvas, opt.step, opt.swatch),
      prefix=opt.prompt_prefix,
      suffix=opt.prompt_suffix,
      save_progress=(opt.save_progress and not opt.nop)
  )
  _CANVAS.show()


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
  parser.add_argument('--prompts',
                      type=argparse.FileType('r', encoding='utf-8'),
                      default='prompts.txt')
  parser.add_argument('--prompt_prefix', type=str, default=None)
  parser.add_argument('--prompt_suffix', type=str, default=None)
  parser.add_argument('--prompt_change_rate', type=int, default=1)

  parser.add_argument('--prompt_strength', type=float, default=_PROMPT_STRENGTH)
  parser.add_argument('--guidance_scale', type=float, default=_GUIDANCE_SCALE)
  parser.add_argument('--steps', type=int, default=_STEPS)

  parser.add_argument('--noise', type=float, default=_NOISE)
  parser.add_argument('--nop', action='store_true', default=_NOP)
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
