"""
Run the Indian Cattle & Buffalo Recognition System
"""

import uvicorn
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Run the FastAPI application."""
    print("\n" + "="*60)
    print("  Indian Cattle & Buffalo Recognition System")
    print("  AI-Powered Breed & Age Identification")
    print("="*60 + "\n")

    print("Starting server...")

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "localhost")

    print(f"  - Web Interface: http://localhost:{port}")
    print(f"  - API Documentation: http://localhost:{port}/docs")
    print(f"  - Admin Panel: http://localhost:{port}/admin")
    print(f"  - (Alt) Web Interface: http://127.0.0.1:{port}")
    print("\nPress Ctrl+C to stop the server.\n")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "").lower() in ("1", "true", "yes"),
        log_level="info"
    )

if __name__ == "__main__":
    main()
