#!/usr/bin/env python3
import argparse
import datetime
import os
import pathlib
import random
import replicate
from PIL import Image
import imagestuff

_NOP = False

_SWATCH = 512, 512
_PROMPT_STRENGTH = 0.7
_GUIDANCE_SCALE = 7.5
_STEPS = 25
_NOISE = 0.05

_CANVAS = None

stamp = datetime.datetime.now().isoformat('T', 'seconds').replace(':', '')

def tup_add(a, b): return tuple(x + y for x, y in zip(a, b))
def tup_sub(a, b): return tuple(x - y for x, y in zip(a, b))
def tup_mul(a, b): return tuple(x * y for x, y in zip(a, b))
def tup_div(a, b): return tuple(x / y if x > 0 else 0 for x, y in zip(a, b))
def tup_idiv(a, b): return tuple(x // y if x > 0 else 0 for x, y in zip(a, b))
def tup_ceil(a): return tuple(round(x + 0.5) for x in a)
def tup_within(a, lo, hi): return tuple((l <= x and x < h) for x, l, h in zip(a, lo, hi))

def mkfilename(suffix):
  return pathlib.Path(f'img_{stamp}_{suffix}.png')

def add_image(image, position=None):
  if image:
    if position is None:
      position = (random.randrange(_CANVAS.size[0] - image.size[0] + 1),
                  random.randrange(_CANVAS.size[1] - image.size[1] + 1))
    # TODO: check overlapping background, and normalise the colours of the
    # incoming image to match what's being replaced.
    _CANVAS.paste(image, position)

def get_image(position):
  box = (*position, *tup_add(position, _SWATCH))
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
  else:
    raise RuntimeError('retries exceeded trying to escape NSFW loop')
  results = [ imagestuff.download(url) for url in results ]
  for image in results:
    imagestuff.log(image)
  return results


def update_canvas(position, prompt):
  start = get_image(position)

  # fully-transparent gives entropy=2.0?
  entropy = start.convert('L').entropy()
  if entropy < 0.05:
    print(f'dropping start images for lack of entropy {entropy}.')
    start = None
    alpha = None
  else:
    start = imagestuff.outpaint(start, extra_noise=_NOISE)
    alpha = start.getchannel("A").convert(mode="L")
    start.putalpha(255)
    start = start.convert('RGB')

  if _NOP:
    result = start.copy() if start else None
  else:
    images = predict(prompt, start, alpha)
    result = images[0]

  add_image(result, position)


def outpaint(start_image, prompt_gen, coord_gen):
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

    imagestuff.save_log(mkfilename(position))
    _CANVAS.save(mkfilename('result'), 'PNG')


def random_coords(area, step, swatch):
  subdiv = tup_ceil(tup_div(tup_sub(area, swatch), step))
  step = tup_idiv(tup_sub(area, swatch), subdiv)

  results = []
  for x_pos in range(subdiv[0] + 1):
    for y_pos in range(subdiv[1] + 1):
      tup = tup_mul((x_pos, y_pos), step)
      results.append(tup)
  random.shuffle(results)
  yield from results


def spiral_coords(area, step, swatch):
  subdiv = tup_ceil(tup_div(tup_sub(area, swatch), step))
  step = tup_idiv(tup_sub(area, swatch), subdiv)

  centre = tup_idiv(subdiv, (2, 2))
  ncentre = tup_sub(subdiv, centre)

  def result(x, y):
    result = tup_add(centre, (x, y))
    if all(tup_within(result, (0, 0), tup_add(subdiv, (1, 1)))):
      yield tup_mul(result, step)

  print(subdiv, step, centre, ncentre)
  yield from result(0, 0)
  for r in range(1, max(ncentre[0], ncentre[1]) + 1):
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
      slow_iter(prompts, opt.prompt_change_rate),
      #random_coords(opt.canvas, opt.step, opt.swatch)
      spiral_coords(opt.canvas, opt.step, opt.swatch)
  )
  _CANVAS.show()


def xy_pair(arg):
  try:
    x, y = arg.split(',')
    return int(x), int(y)
  except ValueError as exc:
    raise argparse.ArgumentTypeError(f'not an int pair: "{arg}"') from exc

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--canvas', type=xy_pair, default=(800,800))
  parser.add_argument('--step', type=xy_pair, default=(288,288))
  parser.add_argument('--swatch', type=xy_pair, default=_SWATCH)
  parser.add_argument('--start_image', type=str, default=None)
  parser.add_argument('--prompts',
                      type=argparse.FileType('r', encoding='utf-8'),
                      default='prompts.txt')
  parser.add_argument('--prompt_change_rate', type=int, default=1)

  parser.add_argument('--prompt_strength', type=float, default=_PROMPT_STRENGTH)
  parser.add_argument('--guidance_scale', type=float, default=_GUIDANCE_SCALE)
  parser.add_argument('--steps', type=int, default=_STEPS)

  parser.add_argument('--noise', type=float, default=_NOISE)
  parser.add_argument('--nop', action='store_true', default=_NOP)
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
