#!/usr/bin/env bash
# Define your list of layers
layers=('spline' 'rbf' 'cheby' 'kaln' 'gram' 'relu' 'fourier' 'mlp' 'stft' 'stftmlp')

# Loop through each layer and execute main.py with the corresponding arguments
for layer in "${layers[@]}"
do
    echo "Running experiment for layer: $layer"

    # Check if the layer needs the special format (first letter uppercase + 'KanLiteDGCNN')
    if [[ "$layer" =~ ^(spline|kaln|gram|relu|cheby|rbf|stft|fourier)$ ]]; then
        exp_name="$(tr '[:lower:]' '[:upper:]' <<< ${layer:0:1})${layer:1}KanLiteDGCNN"  # First letter to uppercase + KanLiteDGCNN
    else
        exp_name="${layer}LiteDGCNN"  # For other layers, just add 'LiteDGCNN'
    fi

    # Execute different commands for 'fourier' and others
    if [ "$layer" = "fourier" ]; then
        python main.py --exp_name "$exp_name" --layer_type "$layer" --batch_size 2 --test_batch_size 1
    else
        python main.py --exp_name "$exp_name" --layer_type "$layer"
    fi
done

