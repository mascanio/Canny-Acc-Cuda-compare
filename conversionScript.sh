#!/bin/bash

convert -type truecolor -units PixelsPerInch -density 72 -compress None -depth 24 $1 BMP3:$2
