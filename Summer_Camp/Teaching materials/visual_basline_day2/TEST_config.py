from my_lib import *


cfg = VisualConfig(random_state=1000)
cfg.to_json("example.json")

print(cfg)
