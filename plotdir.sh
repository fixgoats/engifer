#!/bin/env bash
for filename in $1/*.npz; do
    python plotpsir.py "$filename" "$filename.png"
done
