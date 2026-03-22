#!/bin/bash

# ====== CONFIG ======
LOCAL_FILE="main.pdf"
REMOTE_USER="xfliu"
REMOTE_HOST="www.xfliu.org"
REMOTE_PATH="~/"   # home directory

# ====== CHECK FILE ======
if [ ! -f "$LOCAL_FILE" ]; then
    echo "Error: $LOCAL_FILE not found!"
    exit 1
fi

# ====== SCP ======
echo "Uploading $LOCAL_FILE to $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

scp "$LOCAL_FILE" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}"

# ====== RESULT ======
if [ $? -eq 0 ]; then
    echo "Upload successful."
else
    echo "Upload failed."
fi

ssh www.xfliu.org "cp ~/main.pdf /var/www/xfliu.org/pub/main.pdf"
