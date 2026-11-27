from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.models.domain import Ticket, KnowledgeDoc
from app.schemas.domain import (
    TicketIngest, TicketResponse, CategorizeRequest, CategorizeResponse,
    SuggestReplyRequest, SuggestReplyResponse, KnowledgeDocCreate, KnowledgeDocResponse,
    KBSearchResponse
)
from app.services.llm_provider import llm_service
from app.services.vector_search import mock_text_embedding, cosine_similarity
from app.core.config import settings

api_router = APIRouter()


def _has_meaningful_text(text: str) -> bool:
    """True if the text contains at least one letter."""
    return any(ch.isalpha() for ch in text)


# Ingest Ticket
@api_router.post("/tickets", response_model=TicketResponse, tags=["Tickets"])
def ingest_ticket(ticket_in: TicketIngest, db: Session = Depends(get_db)):
    category = llm_service.categorize_ticket(ticket_in.subject, ticket_in.content)
    reply = llm_service.generate_response(ticket_in.subject, ticket_in.content)

    ticket = Ticket(
        customer_email=ticket_in.customer_email,
        subject=ticket_in.subject,
        content=ticket_in.content,
        category=category,
        ai_suggested_reply=reply
    )
    db.add(ticket)
    db.commit()
    db.refresh(ticket)
    return ticket


@api_router.get("/tickets", response_model=List[TicketResponse], tags=["Tickets"])
def list_tickets(db: Session = Depends(get_db)):
    return db.query(Ticket).all()


# AI Actions
@api_router.post("/ai/categorize", response_model=CategorizeResponse, tags=["AI Services"])
def categorize_text(req: CategorizeRequest):
    cat = llm_service.categorize_ticket("Query", req.text)
    return {"category": cat, "confidence": 0.92}


@api_router.post("/ai/suggest-reply", response_model=SuggestReplyResponse, tags=["AI Services"])
def suggest_reply(req: SuggestReplyRequest):
    reply = llm_service.generate_response(req.subject, req.content)
    return {"suggested_reply": reply, "provider": settings.LLM_PROVIDER}


# Knowledge Base RAG Search
@api_router.post("/knowledge-base", response_model=KnowledgeDocResponse, tags=["Knowledge Base"])
def create_kb_doc(doc_in: KnowledgeDocCreate, db: Session = Depends(get_db)):
    doc = KnowledgeDoc(title=doc_in.title, content=doc_in.content, category=doc_in.category)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


@api_router.get("/knowledge-base/search", response_model=List[KBSearchResponse], tags=["Knowledge Base"])
def search_kb(query: str, db: Session = Depends(get_db)):
    # Reject empty, whitespace-only, or punctuation-only queries
    if not query or not _has_meaningful_text(query):
        raise HTTPException(
            status_code=400,
            detail="Please enter meaningful search keywords",
        )

    query_vec = mock_text_embedding(query)
    docs = db.query(KnowledgeDoc).all()
    results = []

    for doc in docs:
        doc_vec = mock_text_embedding(f"{doc.title} {doc.content}")
        sim = cosine_similarity(query_vec, doc_vec)
        results.append({
            "doc_id": doc.id,
            "title": doc.title,
            "snippet": doc.content[:150] + "...",
            "similarity_score": sim
        })

    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results[:5]