#!/bin/bash

echo "srun --gres=gpu:"$1" --time="$2:00:00" --pty /bin/bash -l"
srun --gres=gpu:"$1" --time="$2:00:00" --pty /bin/bash -l