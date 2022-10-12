import io
import random
import requests
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

import glstuff

_DEBUG_LOG = []

def reset_log():
  global _DEBUG_LOG
  _DEBUG_LOG = []

def log(img):
  _DEBUG_LOG.append(img.copy())

def save_log(filename):
  if not _DEBUG_LOG: return
  width = sum(map(lambda i: i.size[0], _DEBUG_LOG))
  height = max(map(lambda i: i.size[1], _DEBUG_LOG))
  output = Image.new('RGB', (width, height))
  xoff = 0
  for image in _DEBUG_LOG:
    output.paste(image, (xoff, 0))
    xoff += image.size[0]
  output.save(filename)
  reset_log()


def download(url):
  with requests.get(url, stream=True) as link:
    fileobj = io.BytesIO(link.content)
    with Image.open(fileobj, mode='r') as image:
      image.load()
      return image


def match_histogram(image, histogram):
  wrong_hist = image.histogram()

  lut = []
  for p in range(0, len(wrong_hist), 256):
    scale = sum(histogram[p:p+256]) / sum(wrong_hist[p:p+256])
    s = 0
    i = 0
    i0 = 0
    for o in range(256):
      s += wrong_hist[p + o] * scale
      if s > 0:
        i0 = i
        while s > 0:
          s -= histogram[p + i]
          i += 1
      lut.append((i + i0) // 2)
  return image.point(lut)


def area_of_interest(src):
  # TODO: compensate for transparent regions saturating the result
  grey = src.convert("L")
  greyblur = grey.filter(ImageFilter.GaussianBlur(8))
  grey = ImageChops.difference(grey, greyblur)
  grey = ImageOps.equalize(grey)
  box = (0, 0, grey.size[0] - 1, grey.size[1] - 1)
  ImageDraw.Draw(grey).rectangle(box, outline=255, width=1)
  greyblur = grey.filter(ImageFilter.GaussianBlur(64))
  greyblur = ImageOps.equalize(greyblur)
  return greyblur


def outpaint(src, cutout_radius=24, gradient_radius=64, extra_noise=0.03, roi_pinch=(0.3,0.5)):
  aoi = area_of_interest(src)
  alpha = src.getchannel("A").convert(mode="L")
  alpha1 = alpha.filter(ImageFilter.GaussianBlur(gradient_radius))
  alpha2 = alpha.filter(ImageFilter.GaussianBlur(cutout_radius))
  fuzzyalpha = Image.merge("RGBA", (aoi, alpha1, alpha2, alpha))
  log(src)
  log(aoi)

  with open('outpainter.frag', 'rt', encoding='utf-8') as file:
    shader = file.read()
  args = {
      'random': (random.random(), random.random(), random.random(), random.random()),
      'alphapinch': (0.7, 0.8),
      'roipinch': roi_pinch,
      'extra_noise': extra_noise,
      'image_texture': src,
      'alpha_texture': fuzzyalpha,
  }
  return glstuff.shade_it(src.size, shader=shader, shader_args=args)
