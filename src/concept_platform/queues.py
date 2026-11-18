"""Celery queue names used by the three worker pools."""

AGENTS_QUEUE = "agents"
ENGINEERING_QUEUE = "engineering"
DOCUMENTS_QUEUE = "documents"

QUEUES = (AGENTS_QUEUE, ENGINEERING_QUEUE, DOCUMENTS_QUEUE)
