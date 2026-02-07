#!/bin/bash
# Diagnostic script for Codex extension issues in Cursor (WSL2)

echo "=== Codex Extension Diagnostic ==="
echo ""

# Check if we're in WSL2
if grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null ; then
    echo "✓ Running in WSL2"
else
    echo "⚠ Not detected as WSL2"
fi
echo ""

# Check Cursor extension directories
CURSOR_HOME="$HOME/.cursor-server"
echo "Checking Cursor directories..."
echo "Cursor home: $CURSOR_HOME"
echo ""

if [ -d "$CURSOR_HOME" ]; then
    echo "✓ Cursor directory exists"
    
    # Check for Codex/OpenAI extensions
    echo ""
    echo "Installed OpenAI/Codex extensions:"
    find "$CURSOR_HOME/extensions" -maxdepth 1 -type d -iname "*openai*" -o -iname "*codex*" 2>/dev/null | while read dir; do
        if [ -d "$dir" ]; then
            echo "  - $(basename "$dir")"
        fi
    done
    
    # Check extension storage
    echo ""
    echo "Extension storage directories:"
    if [ -d "$CURSOR_HOME/User/globalStorage" ]; then
        find "$CURSOR_HOME/User/globalStorage" -maxdepth 1 -type d -iname "*openai*" -o -iname "*codex*" 2>/dev/null | while read dir; do
            if [ -d "$dir" ]; then
                echo "  - $(basename "$dir")"
                echo "    Size: $(du -sh "$dir" 2>/dev/null | cut -f1)"
            fi
        done
    fi
else
    echo "⚠ Cursor directory not found at $CURSOR_HOME"
fi
echo ""

# Test network connectivity
echo "Testing network connectivity..."
echo ""

# Test OpenAI API endpoint
echo "Testing OpenAI API connectivity:"
if curl -s --max-time 5 -o /dev/null -w "HTTP Status: %{http_code}\n" https://api.openai.com/v1/models 2>/dev/null; then
    echo "✓ Can reach OpenAI API"
else
    echo "✗ Cannot reach OpenAI API (may be firewall/proxy issue)"
fi
echo ""

# Check DNS
echo "DNS Configuration:"
if [ -f /etc/resolv.conf ]; then
    echo "  Resolvers:"
    grep nameserver /etc/resolv.conf | head -3
else
    echo "  ⚠ /etc/resolv.conf not found"
fi
echo ""

# Check for proxy settings
echo "Environment proxy settings:"
if [ -n "$HTTP_PROXY" ] || [ -n "$HTTPS_PROXY" ]; then
    echo "  HTTP_PROXY: ${HTTP_PROXY:-not set}"
    echo "  HTTPS_PROXY: ${HTTPS_PROXY:-not set}"
else
    echo "  No proxy environment variables set"
fi
echo ""

# Check WSL2 network interface
echo "WSL2 Network Interface:"
if ip addr show eth0 &>/dev/null; then
    ip addr show eth0 | grep "inet " | awk '{print "  IP: " $2}'
else
    echo "  ⚠ eth0 interface not found"
fi
echo ""

# Summary
echo "=== Summary ==="
echo "If you see authentication errors, try:"
echo "1. Clear extension cache: rm -rf ~/.cursor-server/extensions/openai.chatgpt-*"
echo "2. Clear storage: rm -rf ~/.cursor-server/User/globalStorage/openai.chatgpt/"
echo "3. Restart Cursor and re-authenticate"
echo ""
echo "If network tests fail, check:"
echo "1. Windows Firewall settings"
echo "2. WSL2 network configuration"
echo "3. VPN/Proxy settings"

















