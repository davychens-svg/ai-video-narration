#!/bin/bash

# Inerbee Video Narrator - Complete Startup Script
# Starts both backend (FastAPI) and frontend (React + Vite)

echo "🚀 Starting Inerbee Video Narrator (Full Stack)..."
echo ""

# Check if virtual environment exists for backend
if [ ! -d "venv" ]; then
    echo "❌ Backend virtual environment not found. Please run ./run.sh first to set up the backend."
    exit 1
fi

# Check if frontend node_modules exists
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $(jobs -p) 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend server
echo "🔧 Starting backend server on http://localhost:8001..."
source venv/bin/activate
cd server
python main.py &
BACKEND_PID=$!
cd ..

# Wait a bit for backend to initialize
sleep 3

# Start frontend development server
echo "🎨 Starting frontend dev server on http://localhost:3001..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Both services started successfully!"
echo ""
echo "📡 Backend (FastAPI):  http://localhost:8001"
echo "🎨 Frontend (React):   http://localhost:3001"
echo ""
echo "💡 Open http://localhost:3001 in your browser to use the app"
echo "🔴 Press Ctrl+C to stop all services"
echo ""

# Wait for all background jobs
wait
