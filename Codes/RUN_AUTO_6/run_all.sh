#!/bin/bash
# run_all.sh — Lance EMG-VBA sur une config choisie puis trace les ratios
#
# Usage :
#   ./run_all.sh                 # menu interactif
#   ./run_all.sh visage          # config par nom (sans .json)
#   ./run_all.sh all_images
#   ./run_all.sh all             # toutes les configs

set -e
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

configs=(configs/*.json)
if [ ${#configs[@]} -eq 0 ]; then
    echo "Erreur: aucun fichier dans configs/*.json"
    exit 1
fi

# Resolution de la selection
selected=()
if [ $# -ge 1 ]; then
    if [ "$1" = "all" ]; then
        selected=("${configs[@]}")
    else
        cand="configs/$1.json"
        if [ -f "$cand" ]; then
            selected=("$cand")
        elif [ -f "$1" ]; then
            selected=("$1")
        else
            echo "Erreur: config '$1' introuvable"
            echo "Disponibles:"
            for c in "${configs[@]}"; do
                echo "  - $(basename "$c" .json)"
            done
            exit 1
        fi
    fi
else
    echo "================================================"
    echo "  Choix de la config"
    echo "================================================"
    i=1
    for c in "${configs[@]}"; do
        echo "  $i) $(basename "$c" .json)"
        i=$((i+1))
    done
    echo "  a) toutes"
    echo ""
    read -p "Selection: " choice
    if [ "$choice" = "a" ] || [ "$choice" = "all" ]; then
        selected=("${configs[@]}")
    elif [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le ${#configs[@]} ]; then
        selected=("${configs[$((choice-1))]}")
    else
        echo "Erreur: selection invalide"
        exit 1
    fi
fi

echo ""
echo "================================================"
echo "  Lancement des benchmarks"
echo "================================================"

for config in "${selected[@]}"; do
    echo ""
    echo ">>> $config"
    python run.py "$config"
done

echo ""
echo "================================================"
echo "  Trace des ratios compares"
echo "================================================"

python plot_ratios.py

echo ""
echo "Termine."
