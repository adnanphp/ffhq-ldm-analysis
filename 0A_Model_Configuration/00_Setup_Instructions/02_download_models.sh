#!/bin/bash
# Download specific models from ommer-lab.com

echo "=== Model Downloader ==="

# Essential models (required for analysis)
download_essential() {
    echo "Downloading essential models..."
    
    # FFHQ model (face generation)
    echo "Downloading FFHQ model (2.1GB)..."
    wget https://ommer-lab.com/files/latent-diffusion/ffhq.zip -P models/ldm/
    unzip models/ldm/ffhq.zip -d models/ldm/ffhq-ldm-vq-4/
    rm models/ldm/ffhq.zip
    
    # Text2Img model
    echo "Downloading Text2Img model (3.6GB)..."
    wget https://ommer-lab.com/files/latent-diffusion/text2img.zip -P models/ldm/
    unzip models/ldm/text2img.zip -d models/ldm/text2img-large/
    rm models/ldm/text2img.zip
    
    echo "Essential models downloaded!"
}

# All available models
download_all() {
    echo "Downloading all available models (20+ GB)..."
    
    models=(
        "ffhq.zip"
        "text2img.zip"
        "celeba.zip"
        "cin.zip"
        "lsun_bedrooms.zip"
        "lsun_churches.zip"
        "inpainting_big.zip"
        "semantic_synthesis256.zip"
        "kl-f4.zip"
        "kl-f8.zip"
        "vq-f4.zip"
        "vq-f8.zip"
    )
    
    for model in "${models[@]}"; do
        echo "Downloading: $model"
        wget https://ommer-lab.com/files/latent-diffusion/${model} -P models/
        
        # Extract (remove .zip extension)
        folder_name="${model%.zip}"
        unzip "models/${model}" -d "models/${folder_name}/"
        rm "models/${model}"
    done
    
    echo "All models downloaded!"
}

# Download specific model
download_specific() {
    echo "Available models:"
    echo "1) ffhq.zip - Face generation (2.1GB)"
    echo "2) text2img.zip - Text to image (3.6GB)"
    echo "3) celeba.zip - CelebA faces (2.1GB)"
    echo "4) lsun_bedrooms.zip - Bedroom scenes (2.1GB)"
    echo "5) inpainting_big.zip - Inpainting (3.1GB)"
    
    read -p "Enter model number [1-5]: " model_num
    
    declare -A model_map=(
        [1]="ffhq.zip"
        [2]="text2img.zip"
        [3]="celeba.zip"
        [4]="lsun_bedrooms.zip"
        [5]="inpainting_big.zip"
    )
    
    model_name="${model_map[$model_num]}"
    if [ -n "$model_name" ]; then
        echo "Downloading: $model_name"
        wget https://ommer-lab.com/files/latent-diffusion/${model_name} -P models/
        
        folder_name="${model_name%.zip}"
        unzip "models/${model_name}" -d "models/${folder_name}/"
        rm "models/${model_name}"
        
        echo "Model downloaded: $folder_name"
    else
        echo "Invalid selection"
    fi
}

# Main menu
echo "Select download option:"
echo "1) Download essential models only (5.7GB)"
echo "2) Download all available models (20+ GB)"
echo "3) Download specific model"
read -p "Choice [1-3]: " choice

case $choice in
    1) download_essential ;;
    2) download_all ;;
    3) download_specific ;;
    *) echo "Invalid choice" ;;
esac
