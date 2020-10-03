#!/bin/bash
docker exec -itd PatchSets tensorboard --logdir=. --host=0.0.0.0 --port=${@-6006}
