#!/bin/bash

# Try to build the database, but don't fail if it errors (e.g., network issues)
echo "Attempting to build ChromaDB index..."
if python indexer.py 2>&1; then
    echo "✓ ChromaDB index built successfully"
else
    echo "⚠ Warning: Could not build ChromaDB index (network/model download issue)"
    echo "  The app will start, but search functionality may be limited."
    # Create empty db directory if it doesn't exist
    mkdir -p ./db
fi

echo "Starting EkoBot API server..."
uvicorn main:app --host 0.0.0.0 --port 7860
