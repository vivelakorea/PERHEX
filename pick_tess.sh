#!/bin/bash
# Generate candidate periodic tessellations; rank them with eval_tess.py and
# copy the best (largest minimum edge) to rve.tess.
cd "$(dirname "$0")"
for id in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  neper -T -n 20 -periodicity all -morpho gg -id $id -o cand_$id
done
