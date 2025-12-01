#!/bin/bash

# Quick test of the NLP-enhanced multi-orchestrator

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Testing Multi-Orchestrator with NLP Understanding        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Build the orchestrator
echo "📦 Building multi-orchestrator..."
go build -o cmd/multi_orchestrator/multi_orchestrator ./cmd/multi_orchestrator
if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi
echo "✅ Build successful"
echo ""

# Run tests
echo "🧪 Running semantic parsing tests..."
go test -v ./cmd/multi_orchestrator -run TestSemanticParsing 2>&1 | grep -E "(Test:|Intent:|Entities:|✅)" | head -20
echo ""

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Example Natural Language Commands                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Try these commands with the orchestrator:"
echo ""
echo "  1. create a webserver with authentication handler"
echo "     → Intent: create_handler"
echo "     → Entities: handler_name='authentication'"
echo ""
echo "  2. I need a database for storing users"
echo "     → Intent: create_database"
echo "     → Entities: database_name='users'"
echo ""
echo "  3. build me a Go API server with JWT"
echo "     → Intent: add_feature"
echo "     → Entities: component='API', feature='JWT'"
echo ""
echo "  4. create handler called payment"
echo "     → Intent: create_handler"
echo "     → Entities: handler_name='payment'"
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Key Features                                              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "✅ Intent Classification - Understands what you want to do"
echo "✅ Entity Extraction - Identifies names, types, parameters"
echo "✅ Template Detection - Recognizes scaffolding patterns"
echo "✅ Semantic Output - Generates structured JSON representation"
echo "✅ Natural Language - Write commands as you would speak"
echo ""
echo "To run the orchestrator interactively:"
echo "  ./cmd/multi_orchestrator/multi_orchestrator"
echo ""
