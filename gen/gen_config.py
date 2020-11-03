# Copyright 2020 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

import os
import sys
import json

style_source_dir = sys.argv[1]
style_images = list()
for x in os.listdir(style_source_dir):
    style_images.append(os.path.join(style_source_dir,x))
print(style_images)

text_file = sys.argv[2]
texts = list(map(lambda s: s.strip(), open(text_file).readlines()))
print(texts)

out_dir = sys.argv[3]
out_config = sys.argv[4]

js_obj = list()

for f in style_images:
    for i, t in enumerate(texts):
        o = os.path.join(out_dir, '%s_%d.png' % (os.path.basename(f[:-4]), i))
        obj = {'style': [f], 'text': [t], 'out': [o]}
        js_obj.append(obj)

json.dump(js_obj, open(out_config, 'w'), indent=4)



