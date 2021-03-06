#!/bin/sh

session="pyai"
# tmux kill-session -t $session
session_exists=$(tmux list-sessions | grep $session)

if [ "$session_exists" = "" ]; then
    tmux new-session -d -s $session -x "$(tput cols)" -y "$(tput lines)"

    tmux rename-window -t 0 "default"
    tmux send-keys -t "default" "conda activate dnn" C-m "clear" C-m "vim" C-m
    tmux split-window -v -p 25
    tmux send-keys -t 1 "cd src" C-m "conda activate dnn" C-m "clear" C-m
    tmux select-pane -t 1
    tmux split-window -h -p 50
    tmux send-keys -t 2 "conda activate dnn" C-m "clear" C-m
    tmux select-pane -t 0
fi

tmux select-window -t $session:0
tmux attach-session -t $session
