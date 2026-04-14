"""
AWS Lambda entry point — wraps the FastAPI app via Mangum.

Deploy with:
    sam build && sam deploy --guided
"""

from mangum import Mangum

from src.main import app

# Mangum translates API Gateway proxy events into ASGI requests
handler = Mangum(app, lifespan="off")
