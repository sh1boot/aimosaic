#!/usr/bin/env python3
from collections import deque
import datetime
import os
import pathlib
import random
import replicate
from PIL import Image
import imagestuff

NOP = False

SWATCH = 512, 512
CANVAS = 768, 768
STEP = 256, 256
PROMPT_STRENGTH = 0.7
GUIDANCE_SCALE = 7.5
STEPS = 50

canvas = Image.new("RGBA", CANVAS)

stamp = datetime.datetime.now().isoformat('T', 'seconds').replace(':', '')

def mkfilename(suffix):
  return pathlib.Path(f'img_{stamp}_{suffix}.png')

def add_image(image, position):
  if image: canvas.paste(image, position)

def predict(prompt, start, alpha):
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
            width=SWATCH[0],
            height=SWATCH[1],
            init_image=startpath,
            mask=alphapath,
            prompt_strength=PROMPT_STRENGTH,
            num_outputs=1,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
            seed=None
        )
        os.remove(startpath)
        os.remove(alphapath)
      else:
        results = model.predict(
            prompt=try_prompt,
            width=SWATCH[0],
            height=SWATCH[1],
            prompt_strength=PROMPT_STRENGTH,
            num_outputs=1,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE_SCALE,
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
  return [ imagestuff.download(url) for url in results ]


def request_image(position, prompt):
  box = (*position, position[0] + SWATCH[0], position[1] + SWATCH[1])

  imagestuff.reset_log()
  start = canvas.crop(box=box)

  # fully-transparent gives entropy=2.0?
  entropy = start.convert('L').entropy()
  if entropy < 0.05:
    print(f'dropping start images for lack of entropy {entropy}.')
    start = None
    alpha = None
  else:
    start = imagestuff.outpaint(start)
    imagestuff.log(start)
    alpha = start.getchannel("A").convert(mode="L")
    imagestuff.log(alpha)
    start.putalpha(255)
    start = start.convert('RGB')

  if NOP:
    result = start.copy() if start else None
  else:
    images = predict(prompt, start, alpha)
    for image in images:
      imagestuff.log(image)
    result = images[0]

  add_image(result, position)

  imagestuff.save_log(mkfilename(position))


def outpaint(image, prompt_gen, coords, prompt_len=1):
  window = deque(maxlen=prompt_len)
  while len(window) < window.maxlen:
    window.append(next(prompt_gen).strip())

  if image: add_image(image, next(coords))
  for coord in coords:
    while True:
      window.append(next(prompt_gen).strip())
      prompt = ' '.join(window)
      print(f'{coord[0]},{coord[1]} prompt: {prompt}')
      try:
        request_image(coord, prompt)
        break
      except RuntimeError:
        print('rotating prompt to try again')

    canvas.save(mkfilename('result'), 'PNG')

def coord_gen(step):
  subdiv_x = (CANVAS[0] - SWATCH[0]) // step[0]
  subdiv_y = (CANVAS[1] - SWATCH[1]) // step[1]
  print('subdiv:', subdiv_x, subdiv_y)
  step = ((CANVAS[0] - SWATCH[0]) // subdiv_x, (CANVAS[1] - SWATCH[1]) // subdiv_y)
  print('step:', step)
  results = []
  for x_pos in range(subdiv_x + 1):
    for y_pos in range(subdiv_y + 1):
      tup = (x_pos * step[0], y_pos * step[1])
      results.append(tup)
  random.shuffle(results)
  yield from results

with open('sometext.txt', 'rt', encoding='utf-8') as file:
  # for i in range(random.randrange(30)): next(file)
  outpaint(None, file, coord_gen(STEP))
  canvas.show()
