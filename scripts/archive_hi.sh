#!/bin/bash
cd /Users/anrunw/Documents/phat-jet-hw
while [ ! -f STOP_ARCH ]; do
  for d in pareto_qat_n64_w1 pareto_qat_n64_w2 pareto_qat_n64_w3; do
    [ -d "$d" ] || continue
    for f in "$d"/*.keras; do
      [ -e "$f" ] || continue
      a=$(echo "$f" | sed -n 's/.*val_acc=\([0-9.]*\).*/\1/p')
      awk "BEGIN{exit !($a >= 0.8150)}" && cp -n "$f" "archive_hi/$(basename $d)_$(basename $f)" 2>/dev/null
    done
  done
  sleep 60
done
