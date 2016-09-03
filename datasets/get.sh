#!/bin/bash

ME=${1:-cympfh}

id-of() {
    twurl "/1.1/statuses/user_timeline.json?count=1&screen_name=$1" | jq -r '.[0].user.id_str'
}

ME=`id-of $ME`

friends() {
    twurl "/1.1/friends/ids.json?user_id=$1&count=5000&stringify_ids=true" | jq -r '.ids[]'
}

icon() {
    # the profile image of USER_ID
    twurl "/1.1/statuses/user_timeline.json?count=1&user_id=$1" | jq -r '.[0].user.profile_image_url'
}

for id in `friends $ME`; do
    wget -nc `icon $id`
    for id2 in `friends $id`; do
        wget -nc `icon $id2`
        sleep 10
    done
    echo -n .
    sleep 120
done
