import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os


class WordImageGenerator(object):
    def __init__(self, fontList_path, width_whole, width_single, height):

        with open(fontList_path) as font_f:
            self.fontList = [line.rstrip() for line in font_f]

        self.root = os.path.dirname(fontList_path)
        self.width_whole = width_whole
        self.width_single = width_single
        self.height = height
        self.charset = np.array(list("_0123456789abcdefghijklmnopqrstuvwxyz"))

    def __call__(self, target, is_random=True):

        target = target.cpu().detach().numpy()
        target_idx = target < len(self.charset)
        out_list = []
        text_target_cache = np.unique(np.array(["".join(self.charset[target_i[target_idx_i]]) for target_i, target_idx_i in zip(target, target_idx)]))

        for idx, (target_i, target_idx_i) in enumerate(zip(target, target_idx)):

            text_real = "".join(self.charset[target_i[target_idx_i]])
            if len(text_target_cache) == 1 and text_real == '':
                text_fake = text_real
            elif len(text_target_cache) == 1 and text_real != '': 
                text_fake = text_real[::-1]
            else:
                text_fake = random.choice(text_target_cache[text_target_cache!=text_real])

            if is_random:
                # font_index = random.sample(range(0, len(self.fontList - 1)),
                #                            fontNum)
                font_index = random.randint(0, len(self.fontList) - 1)
            else:
                # font_index = list(range(0, fontNum))
                font_index = 0

            out_real = self.compact_lower_upper(text_real, os.path.join(self.root, self.fontList[font_index]))
            out_fake = self.compact_lower_upper(text_fake, os.path.join(self.root, self.fontList[font_index]))
            out_list.append(out_real)
            out_list.append(out_fake)

        return np.stack(out_list, 0)

    def draw_word(self, text, fontName, lower):

        W_whole, H_whole = (720, 32)
        W, H = (32, 32)
        font = ImageFont.truetype(fontName,
                                  self.width_single,
                                  encoding='utf-8')
        #dst = Image.new('1', (W * len(text), H))
        dst = Image.new('1', (self.width_whole, self.height))
        for idx, char in enumerate(text):
            # image = Image.new("1", (self.width_single, self.height), "Black")
            image = Image.new("1", (self.width_single, self.height), "White")
            draw = ImageDraw.Draw(image)
            if lower == True:
                offset_w, offset_h = font.getoffset(char.lower())
                w, h = draw.textsize(char.lower(), font=font)
                pos = ((self.width_single - w - offset_w) / 2,
                       (self.height - h - offset_h) / 2)
                # draw.text(pos, char.lower(), "White", font=font)
                draw.text(pos, char.lower(), "Black", font=font)
            else:
                offset_w, offset_h = font.getoffset(char.upper())
                w, h = draw.textsize(char.upper(), font=font)
                pos = ((self.width_single - w - offset_w) / 2,
                       (self.height - h - offset_h) / 2)
                draw.text(pos, char.upper(), "White", font=font)
            dst.paste(image, ((idx * self.width_single), 0))

        dst_np = np.asarray(dst, dtype=np.int8)

        dst.save("./tmp/" + ''.join([i for i in text]) + str(lower) + ".png")

        return dst_np

    def compact_lower_upper(self, text, fontName):

        dst_np_lower = self.draw_word(text, fontName, lower=True)
        dst_np_upper = self.draw_word(text, fontName, lower=False)

        return np.stack([dst_np_lower, dst_np_upper], 0)
        #return np.expand_dims(dst_np_lower, axis=0)