#!/bin/sh
# launcher.sh

cd /
amixer scontrols
amixer sset 'PCM' 100%
cd home/pi/BirdBeGone
sudo python repel_birds.py
cd /