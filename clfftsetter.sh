#! /bin/bash

if which apt-get > /dev/null; then
    sudo apt-get update
    sudo apt-get install libclfft-dev
elif which dnf > /dev/null; then
    sudo dnf install clfft-devel

elif which pacman > /dev/null; then
    if which yay > /dev/null; then
        echo "yay found"
    else
        echo "Installing YAY for clfft"
        sudo pacman -S --needed git base-devel && git clone https://aur.archlinux.org/yay.git && cd yay && makepkg -si
    fi
    yay -S clfft
fi
