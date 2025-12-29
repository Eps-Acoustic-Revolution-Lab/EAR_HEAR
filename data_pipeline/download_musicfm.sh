#!/bin/bash

# 检查输入参数
if [ $# -ne 1 ]; then
    echo "Usage: $0 <musicfm_checkpoint_path>"
    exit 1
fi


MUSICFM_CKPT_PATH=$1
mkdir -p ${MUSICFM_CKPT_PATH} > /dev/null 2>&1
# download MusicFM model
download_if_needed() {
    local file_path=$1
    local expected_md5=$2
    local download_url=$3
    
    local need_download=1
    
    if [ -f "$file_path" ]; then
        # 计算文件的MD5值
        local actual_md5=$(md5sum "$file_path" | cut -d' ' -f1)
        if [ "$actual_md5" = "$expected_md5" ]; then
            echo "File $file_path exists and MD5 matches, skipping download."
            need_download=0
        else
            echo "File $file_path exists but MD5 mismatch (expected: $expected_md5, actual: $actual_md5), re-downloading..."
        fi
    else
        echo "File $file_path does not exist, downloading..."
    fi
    
    if [ $need_download -eq 1 ]; then
        curl -L "$download_url" -o "$file_path"
        # 验证下载后的MD5值
        local downloaded_md5=$(md5sum "$file_path" | cut -d' ' -f1)
        if [ "$downloaded_md5" = "$expected_md5" ]; then
            echo "Successfully downloaded and verified $file_path"
        else
            echo "Warning: MD5 mismatch after download (expected: $expected_md5, actual: $downloaded_md5)"
            exit 1
        fi
    fi
}

download_if_needed "${MUSICFM_CKPT_PATH}/msd_stats.json" \
    "75ab2e47b093e07378f7f703bdb82c14" \
    "${HF_ENDPOINT:-https://huggingface.co}/minzwon/MusicFM/resolve/main/msd_stats.json"

download_if_needed "${MUSICFM_CKPT_PATH}/pretrained_msd.pt" \
    "df930aceac8209818556c4a656a0714c" \
    "${HF_ENDPOINT:-https://huggingface.co}/minzwon/MusicFM/resolve/main/pretrained_msd.pt"
