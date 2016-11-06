#!/bin/bash

makeplot() {

    TMP1=`mktemp`
    TMP2=`mktemp`

    cat result/log | jq '.[]."main/loss_decode"' >$TMP1
    cat result/log | jq '.[]."main/loss_kl"' >$TMP2

    gnuplot <<EOF
    set terminal png;
    set output 'plot.png';
    set grid;
    filter(x)=(floor(x/10)*10);
    plot "$TMP1" u (filter(\$0)):1 smooth unique axis x1y1 title "decode loss", \
         "$TMP2" u (filter(\$0)):1 smooth unique axis x1y2 title "KL loss"
EOF
    rm $TMP1 $TMP2
}

while :; do

    tw -f ae.png >/dev/null
    tw -f rand.png >/dev/null
    cat result/log | jq -cM '.[]' | tail -1 | tw -
    makeplot && tw -f plot.png >/dev/null

    sleep $(hour 3)
done
