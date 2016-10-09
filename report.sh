#!/bin/bash

while :; do

    AE=$( \ls -1 ae.*.png | tail -1 )
    RA=$( \ls -1 rand*.png | tail -1 )

    echo $AE $RA
    tw-icon $AE >/dev/null
    tw -f $RA >/dev/null

    sleep 600

done
