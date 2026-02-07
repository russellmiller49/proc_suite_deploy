#!/bin/bash
# Script to update Node.js in all conda environments

# Target Node.js version (20.x LTS recommended for Gemini CLI compatibility)
NODEJS_VERSION="20"

# Get list of all conda environments
ENVS=$(conda env list | grep -v "^#" | grep -v "^$" | awk '{print $1}' | grep -v "^$")

echo "Updating Node.js to version ${NODEJS_VERSION}.x in all conda environments..."
echo ""

for env in $ENVS; do
    # Skip if environment name is empty
    [ -z "$env" ] && continue
    
    echo "Checking environment: $env"
    
    # Check if nodejs is installed in this environment
    if conda list -n "$env" nodejs 2>/dev/null | grep -q "^nodejs"; then
        CURRENT_VERSION=$(conda list -n "$env" nodejs 2>/dev/null | grep "^nodejs" | awk '{print $2}')
        echo "  Found nodejs $CURRENT_VERSION"
        echo "  Updating to nodejs ${NODEJS_VERSION}..."
        
        # Update nodejs in this environment
        conda install -n "$env" -y -c conda-forge "nodejs>=${NODEJS_VERSION}" 2>&1 | tail -3
        
        # Verify new version
        NEW_VERSION=$(conda list -n "$env" nodejs 2>/dev/null | grep "^nodejs" | awk '{print $2}')
        echo "  New version: $NEW_VERSION"
    else
        echo "  No nodejs installed, skipping"
    fi
    echo ""
done

echo "Done! To verify, run: conda list -n <env_name> nodejs"

