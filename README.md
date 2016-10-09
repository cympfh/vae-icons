# Variational Autoencoder on Twitter-profile-images

## Requirements

- twurl
- jq
- Python3
    - chainer, PIL

```bash
apt-get install jq
git clone git@github.com:twitter/twurl
pip3 install chainer Pillow
```

## datasets

```bash
mkdir datasets
cd datasets
bash ../get.sh $TWITTER_SCREEN_NAME
```

## vae.py

VAE learning and saving the generatege images

```bash
./vae.py --gpu 0
```
