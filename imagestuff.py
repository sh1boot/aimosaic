import io
import math
import random
import requests
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

import glstuff


class Point(tuple):
  def __new__(cls, x, y=None):
    if y is None:
      if isinstance(x, tuple):
        return super().__new__(cls, x)
      return super().__new__(cls, (x, x))
    return super().__new__(cls, (x, y))

  @property
  def x(self):
    return self[0]
  @property
  def y(self):
    return self[1]

  @classmethod
  def xymap(cls, function, *args):
    return cls(
        function(*(a.x if isinstance(a, cls) else a for a in args)),
        function(*(a.y if isinstance(a, cls) else a for a in args))
    )

  def _op(self, b, f): return type(self).xymap(f, self, type(self)(b))
  def __add__(self, b): return self._op(b, lambda x, y: x + y)
  def __sub__(self, b): return self._op(b, lambda x, y: x - y)
  def __mul__(self, b): return self._op(b, lambda x, y: x * y)
  def __truediv__(self, b): return self._op(b, lambda x, y: x / y if x else 0)
  def __floordiv__(self, b): return self._op(b, lambda x, y: x // y if x else 0)
  def __and__(self, b): return self._op(b, lambda x, y: x & y)
  def __or__(self, b): return self._op(b, lambda x, y: x | y)
  def __lt__(self, b): return self._op(b, lambda x, y: x < y)
  def __le__(self, b): return self._op(b, lambda x, y: x <= y)
  def __gt__(self, b): return self._op(b, lambda x, y: x > y)
  def __ge__(self, b): return self._op(b, lambda x, y: x >= y)
  def __eq__(self, b): return self._op(b, lambda x, y: x == y)
  def __ne__(self, b): return self._op(b, lambda x, y: x != y)
  def __bool__(self): raise TypeError(f'use any() or all() to cast {type(self)} to bool')
  def ceil(self): return type(self)(*(math.ceil(z) for z in self))
  def floor(self): return type(self)(*(math.floor(z) for z in self))

class Size(Point):
  @property
  def width(self): return self[0]
  @property
  def height(self): return self[1]

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
  for plane in range(0, len(wrong_hist), 256):
    scale = (sum(histogram[plane:plane+256]) /
             sum(wrong_hist[plane:plane+256]))
    accum = 0
    i = 0
    i_start = 0
    for o in range(256):
      accum += wrong_hist[plane + o] * scale
      if accum > 0:
        i_start = i
        while accum > 0:
          accum -= histogram[plane + i]
          i += 1
      lut.append((i + i_start) // 2)
  return image.point(lut)


def area_of_interest(src):
  # TODO: compensate for transparent regions saturating the result
  grey = src.convert("L")
  greyblur = grey.filter(ImageFilter.GaussianBlur(8))
  grey = ImageChops.difference(grey, greyblur)
  grey = ImageOps.equalize(grey)
  box = (*(0, 0), *(Size(grey.size) - 1))
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
