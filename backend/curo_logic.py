import os
import requests
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser, PydanticOutputParser

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

from langchain_text_splitters import RecursiveCharacterTextSplitter

from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Any, Dict

from neo4j import GraphDatabase
import numpy as np
import io
import json
import re
import hashlib
from pathlib import Path
from scipy.io import wavfile
from transformers import pipeline

# --- Pydantic Data Models ---

class TriageInteraction(BaseModel):
    question: Optional[str] = Field(None, description="The next diagnostic question for the patient.")
    options: Optional[List[str]] = Field(None, description="3-4 clinical selection options for the user.")
    finished: bool = Field(False, description="Whether the clinical interview is complete.")
    summary: Optional[str] = Field(None, description="A professional clinical summary of the interview history.")

    @model_validator(mode='before')
    @classmethod
    def ensure_valid_triage_state(cls, values):
        if isinstance(values, dict):
            finished = values.get('finished', False)
            if not finished:
                # Ensure options exist
                options = values.get('options')
                if not options or len(options) == 0:
                    values['options'] = ["Other (please specify)"]
                # Ensure question exists
                if not values.get('question'):
                    values['question'] = "Thank you for that information. Could you tell me a bit more about what you're experiencing?"
            else:
                # Ensure summary exists if finished
                if not values.get('summary'):
                    values['summary'] = "Thank you for sharing. I've gathered enough information to proceed."
        return values

class ClinicalRelationship(BaseModel):
    subject: str = Field(description="Source entity (e.g., 'Type 2 Diabetes', 'Headache', 'Metformin'). KEEP IT VERY SHORT (1-3 words).")
    relation: str = Field(description="Relationship (e.g., CAUSES, TREATED_WITH, DIAGNOSED_BY, PRESENTS_WITH, INCREASES_RISK).")
    object: str = Field(description="Target entity (e.g., 'Insulin Resistance', 'Ibuprofen', 'HbA1c'). KEEP IT VERY SHORT.")

class KnowledgeGraphResult(BaseModel):
    topic: str = Field(description="The central anchor disease or condition.")
    relationships: List[ClinicalRelationship] = Field(description="A list of clinical logic triplets extracted from the text.")

# --- Parsers ---
triage_parser = PydanticOutputParser(pydantic_object=TriageInteraction)
kg_parser = PydanticOutputParser(pydantic_object=KnowledgeGraphResult)

# whisper_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en", device="cpu")
# Lazy loading to save memory during startup
_asr_pipe = None

def get_asr_pipe():
    global _asr_pipe
    if _asr_pipe is None:
        print("[*] Loading Whisper ASR Pipeline...")
        _asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")
    return _asr_pipe

# Load environment variables
load_dotenv()

# Initialize Core Models globally for the module with robust FALLBACKS for Rate Limits (429)
primary_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
fallback_llm_1 = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0.1)
fallback_llm_2 = ChatGroq(model_name="gemma2-9b-it", temperature=0.1)
fallback_llm_3 = ChatGroq(model_name="llama3-70b-8192", temperature=0.1)
llm = primary_llm.with_fallbacks([fallback_llm_1, fallback_llm_2, fallback_llm_3])

# Specialized instance for Triage with JSON Mode (with fallbacks)
triage_primary = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3, model_kwargs={"response_format": {"type": "json_object"}})
triage_fallback_1 = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0.3, model_kwargs={"response_format": {"type": "json_object"}})
triage_fallback_2 = ChatGroq(model_name="gemma2-9b-it", temperature=0.3, model_kwargs={"response_format": {"type": "json_object"}})
triage_llm = triage_primary.with_fallbacks([triage_fallback_1, triage_fallback_2])

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def fetch_and_rank_openalex(query_terms: list, email="SASRbros0705@gmail.com") -> list[Document]:
    """Fetches abstracts for ALL differential diagnoses and aggregates them."""
    all_documents = []
    seen_titles = set()

    for term in query_terms[:3]:
        concept_id = None
        try:
            resp = requests.get(
                "https://api.openalex.org/concepts",
                params={"search": term, "mailto": email},
                timeout=10,
            ).json()
            if resp.get("results"):
                concept_id = resp["results"][0]["id"]
        except Exception:
            continue

        if not concept_id:
            continue

        try:
            params = {
                "filter": (
                    f"concepts.id:{concept_id},"
                    "primary_location.source.type:journal,"
                    "is_paratext:false,"
                    "has_abstract:true,"
                    "from_publication_date:2020-01-01"
                ),
                "search": term,
                "sort": "relevance_score:desc",
                "per_page": 20,
                "mailto": email,
            }
            works = requests.get("https://api.openalex.org/works", params=params, timeout=15).json().get("results", [])
        except Exception:
            works = []

        def calculate_authority_score(w):
            title = (w.get('title') or '').lower()
            score = 0
            if any(k in title for k in ["guideline", "consensus", "practice parameter"]): score += 10000
            elif any(k in title for k in ["meta-analysis", "systematic review"]): score += 5000

            pub_date = w.get('publication_date') or ''
            if "2025" in pub_date or "2026" in pub_date: score += 1000

            if "covid" in title or "sars-cov-2" in title or "pandemic" in title:
                if "covid" not in term.lower() and "sars" not in term.lower():
                    score -= 50000

            score += (w.get('cited_by_count') or 0) / 10
            return score

        ranked_works = sorted(works, key=calculate_authority_score, reverse=True)

        docs_added_for_term = 0
        for work in ranked_works:
            if docs_added_for_term >= 3: 
                break

            title = work.get("title")
            if not title or title.lower() in seen_titles: continue
            seen_titles.add(title.lower())

            url = work.get("doi") or work.get("id") or ""

            abs_index = work.get("abstract_inverted_index")
            if not abs_index: continue

            words = sorted([(pos, word) for word, positions in abs_index.items() for pos in positions])
            content = " ".join([w[1] for w in words])

            if len(content) < 150: continue

            trimmed = content[:1500]
            combined_text = f"Diagnosis Analyzed: {term}\nTitle: {title}\nAbstract: {trimmed}"

            all_documents.append(Document(
                page_content=combined_text,
                metadata={"topic": term, "title": title, "url": url}
            ))
            docs_added_for_term += 1

    return all_documents


def update_neo4j_graph(kg_data: dict):
    """Anchors clinical entities to a central Topic node in Neo4j."""
    topic_name = kg_data.get("topic", "General Medical Context")
    relationships = kg_data.get("Relationships", [])
    
    if not relationships:
        return "No relationships to map."
        
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")

    try:
        with GraphDatabase.driver(uri, auth=(user, pwd)) as driver:
            with driver.session() as session:
                for rel in relationships:
                    # Sanitize relationship type for Pure Cypher (No spaces, Uppercase)
                    rel_type = str(rel.get("relation", "PRESENTS_WITH")).replace(" ", "_").upper()
                    
                    # Construct native Cypher query without APOC
                    query = f"""
                        MERGE (t:Topic {{name: $topic}})
                        MERGE (s:Entity {{name: $subj}})
                        MERGE (o:Entity {{name: $obj}})
                        MERGE (s)-[:BELONGS_TO]->(t)
                        MERGE (o)-[:BELONGS_TO]->(t)
                        MERGE (s)-[r:{rel_type}]->(o)
                        RETURN r
                    """
                    session.run(query, topic=topic_name, subj=rel["subject"], obj=rel["object"])
                    
        return f"KG updated for topic: {topic_name}"
    except Exception as e:
        # Check for specific Neo4j Forbidden/Security/ReadOnly errors to fail silently
        err_str = str(e).lower()
        if "forbidden" in err_str or "security" in err_str or "read-only" in err_str:
            print(f"[!] Neo4j Write Restricted (Aura Restriction): {str(e)[:100]}")
            return "KG Update Skipped (Standard write attempt failed)"
        return f"Neo4j Error: {str(e)}"


def read_neo4j_graph(topic_name: str) -> str:
    """Retrieves clinical relationships for a specific Topic anchor."""
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")
    
    try:
        with GraphDatabase.driver(uri, auth=(user, pwd)) as driver:
            with driver.session() as session:
                query = """
                MATCH (e1:Entity)-[r]->(e2:Entity)
                MATCH (e1)-[:BELONGS_TO]->(t:Topic)
                WHERE t.name CONTAINS $name
                RETURN e1.name AS subject, type(r) AS relation, e2.name AS object
                LIMIT 30
                """
                result = session.run(query, name=topic_name)
                relationships = [f"- {record['subject']} {record['relation']} {record['object']}" for record in result]

                if not relationships:
                    return f"No clinical logic stored for {topic_name}."
                return f"Knowledge Graph Context:\n" + "\n".join(relationships)
    except Exception as e:
        return f"Neo4j Read Error: {str(e)}"


def build_local_graph_data(kg_data: dict) -> dict:
    """Builds graph visualization data locally from the extracted KG, no Neo4j dependency."""
    topic = kg_data.get("topic", "Unknown")
    relationships = kg_data.get("Relationships", [])
    
    if not relationships:
        return {"nodes": [], "links": []}
    
    nodes = set()
    links = []
    
    # Add the central topic node
    nodes.add(topic)
    
    for rel in relationships:
        subj = rel.get("subject", "").strip().title()
        obj = rel.get("object", "").strip().title()
        relation = rel.get("relation", "RELATED_TO").strip()
        if subj != obj and subj and obj:
            nodes.add(subj)
            nodes.add(obj)
            links.append({"source": subj, "target": obj, "label": relation.replace(" ", "_").upper()})
    
    # Deduplicate links
    seen_links = set()
    unique_links = []
    for link in links:
        key = (link["source"], link["target"], link["label"])
        if key not in seen_links:
            seen_links.add(key)
            unique_links.append(link)
    
    nodes_list = [{"id": node, "label": node} for node in nodes]
    return {"nodes": nodes_list, "links": unique_links}


def get_neo4j_graph_data(topic_name: str) -> dict:
    """Retrieves clinical relationships for graph visualization (fallback to Neo4j if available)."""
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")
    
    try:
        with GraphDatabase.driver(uri, auth=(user, pwd)) as driver:
            with driver.session() as session:
                query = """
                MATCH (e1:Entity)-[r]->(e2:Entity)
                MATCH (e1)-[:BELONGS_TO]->(t:Topic)
                WHERE t.name CONTAINS $name AND type(r) <> 'BELONGS_TO'
                RETURN e1.name AS subject, type(r) AS relation, e2.name AS object
                """
                result = session.run(query, name=topic_name)
                
                nodes = set()
                links = []
                for record in result:
                    subj = record['subject']
                    rel = record['relation']
                    obj = record['object']
                    
                    nodes.add(subj)
                    nodes.add(obj)
                    links.append({"source": subj, "target": obj, "label": rel})
                
                nodes_list = [{"id": node, "label": node} for node in nodes]
                
                return {"nodes": nodes_list, "links": links}
    except Exception as e:
        return {"nodes": [], "links": []}


def process_curo_query(user_query: str, demography: dict = None, user_id: str = None) -> dict:
    """The main orchestration function modified to return a structured dict for the API."""
    
    # Pre-define some placeholders
    clinical_pathways = "None available for this query."
    
    print(f"--- Processing Query: {user_query} ---\n")
    if demography:
        print(f"[*] Structured Vitals: {demography}")

    # Auto-retrieve health records context if user_id is provided
    local_context = None
    if user_id:
        try:
            local_context = get_health_records_context(user_query, user_id)
            if local_context:
                print(f"[*] Auto-retrieved health records context for user {user_id[:8]}... ({len(local_context)} chars)")
        except Exception as e:
            print(f"[!] Health records retrieval failed: {e}")

    # Build demographic prompt injection
    demo_injection = ""
    if demography and any(demography.values()):
        parts = []
        for k, v in demography.items():
            if v:
                # Map camelCase to friendly display names or Uppercase
                name = k.upper()
                if k == 'spo2': name = 'SPO2 (%)'
                if k == 'temp': name = 'TEMP (°C)'
                if k == 'respRate': name = 'RESPIRATORY RATE'
                parts.append(f"{name}: {v}")
        demo_injection = "\n!!! EXPLICIT PATIENT DEMOGRAPHICS & VITALS !!!\n" + " | ".join(parts) + "\n(You MUST factor these exact objective vitals heavily into your DDx generation. If BP, Heart Rate, SpO2, or Temp is critical, prioritize emergency diagnoses).\n"

    # 1. Entity Extraction (UPDATED PROMPT)
    entity_prompt = PromptTemplate.from_template(
        "ROLE: You are an Expert Medical Diagnostician.\n"
        "TASK: Analyze the following patient presentation and generate a standard Differential Diagnosis (DDx).\n"
        "Output a JSON array of up to three standard, well-recognized medical conditions that best explain the symptoms.\n"
        "CRITICAL RULES:\n"
        "- DIRECT MENTIONS: If the user explicitly states they have, were diagnosed with, or are suffering from a specific disease/condition, you MUST include that exact disease as the MOST LIKELY diagnosis (the very first item in the array).\n"
        "- Use standard medical disease names only (e.g., 'Systemic Sclerosis', 'Myasthenia Gravis', 'Achalasia').\n"
        "- DO NOT use acronyms in parentheses (e.g., use 'Multiple Sclerosis' not 'Multiple Sclerosis (MS)').\n"
        "- Provide base English terminology without special accented characters (e.g., 'Guillain Barre Syndrome').\n"
        "- ACUITY FILTER: If the query contains 'Red Flag' indicators (e.g., tearing pain, asymmetric blood pressure, saddle anesthesia, sudden focal neurological deficits), you MUST prioritize high-risk, time-sensitive emergencies even if they are statistically rare.\n"
        "- DIFFERENTIATION: Ensure the three diagnoses represent distinct clinical possibilities to allow for broad evidence retrieval.\n"
        "- The MOST LIKELY or MOST CRITICAL diagnosis MUST be the first item in the array.\n"
        "- Output ONLY a valid JSON array of strings.\n\n"
        "QUERY: {query}\n{demo_injection}"
    )
    entity_chain = entity_prompt | llm | JsonOutputParser()
    try:
        clinical_terms = entity_chain.invoke({"query": user_query, "demo_injection": demo_injection})
    except Exception:
        clinical_terms = [user_query]

    print(f"[*] DDx Extracted: {clinical_terms}")

    # 2. Fetch Evidence (UPDATED FALLBACK DICT)
    documents = fetch_and_rank_openalex(clinical_terms)
    
    # Inject Health Records Context if available
    if local_context:
        local_doc = Document(
            page_content=f"PATIENT-PROVIDED CLINICAL DOCUMENTATION:\n{local_context}",
            metadata={"topic": "Patient Records", "source": "Health Records Vault", "url": ""}
        )
        documents = [local_doc] + documents
        print(f"[*] Injected health records context ({len(local_context)} chars)")

    if not documents:
        return {
            "response": "No clinical literature found for the specified symptoms.", 
            "extracted_ddx": clinical_terms,
            "winning_diagnosis": "None",
            "abstracts": [],
            "graph_data": {"nodes": [], "links": []}
        }

    # 3. Semantic Firewall
    vectorstore = Chroma.from_documents(documents, embeddings)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    mqr = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.15)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=mqr
    )

    compressed_docs = compression_retriever.invoke(user_query)

    # UPDATED FALLBACK DICT
    if not compressed_docs:
        vectorstore.delete_collection()
        return {
            "response": "No highly relevant literature matched the specific symptom profile.", 
            "extracted_ddx": clinical_terms,
            "winning_diagnosis": "None",
            "abstracts": [],
            "graph_data": {"nodes": [], "links": []}
        }

    print(f"[*] Firewall passed {len(compressed_docs)} highly relevant documents.")

    # 4. LLM Triage Router
    available_topics = list(set([d.metadata["topic"] for d in compressed_docs]))
    if len(available_topics) == 1:
        winning_topic = available_topics[0]
        print(f"[*] Winning Diagnosis (Default): {winning_topic}")
    else:
        triage_prompt = PromptTemplate.from_template(
            "ROLE: You are an Emergency Medicine Attending Physician (Master Diagnostician).\n"
            "TASK: Thoroughly evaluate the patient's exact symptoms and their TIMING to triage the MOST ACCURATE and CRITICAL diagnosis from the 'Available Diagnoses' list.\n"
            "CRITICAL RULES:\n"
            "1. ACUITY TRUMPS PREVALENCE: If the patient's symptoms suggest an immediate life-threatening emergency (e.g., sudden onset over minutes/hours, cold sweat, diaphoresis, radiating pain, tearing sensation), you MUST select the acute emergency from the list (e.g., Myocardial Infarction, Aortic Dissection, DKA) over chronic/slow-developing conditions (e.g., Giant Cell Arteritis).\n"
            "2. COMPLETE OVERLAP: The diagnosis chosen MUST accurately match the symptom timeline, patient age, and presentation severity.\n"
            "3. OUTPUT FORMAT: You must output EXACTLY ONE diagnosis from the 'Available Diagnoses' list. Do NOT add periods, markdown, or any explanation.\n\n"
            "SYMPTOMS: {query}\n{demo_injection}\n"
            "AVAILABLE DIAGNOSES: {topics}\n"
            "SINGLE BEST DIAGNOSIS:"
        )
        triage_chain = triage_prompt | llm | StrOutputParser()
        try:
            llm_choice = triage_chain.invoke({
                "query": user_query,
                "demo_injection": demo_injection,
                "topics": ", ".join(available_topics)
            }).strip()
            winning_topic = llm_choice if llm_choice in available_topics else compressed_docs[0].metadata["topic"]
        except Exception:
            winning_topic = compressed_docs[0].metadata["topic"]
            
        print(f"[*] Winning Diagnosis (LLM Selected): {winning_topic}")

    # 4.5. Real-time Clinical Pathways (Scraping Simulation)
    if winning_topic != "None":
        try:
            print(f"[*] Fetching real-time clinical guidelines for {winning_topic}...")
            # Use LLM to synthesize guidelines from general knowledge/search
            pathway_prompt = PromptTemplate.from_template(
                "ROLE: You are a Clinical Research Assistant.\n"
                "TASK: Synthesize the standard treatment protocol and diagnostic pathway for {topic} based on current clinical guidelines (e.g., NICE, CDC, NIH).\n"
                "Include:\n"
                "- First-line medications/interventions\n"
                "- Required confirmatory tests\n"
                "- Red flags to monitor\n"
                "Keep it concise (2-3 paragraphs)."
            )
            pathway_chain = pathway_prompt | llm | StrOutputParser()
            clinical_pathways = pathway_chain.invoke({"topic": winning_topic})
        except Exception as e:
            print(f"[!] Guideline retrieval failed: {e}")
            clinical_pathways = "Automated guideline retrieval failed."

    # 5. Graph Architect - Revamped for Better Linkage and Lower Token Usage
    kg_prompt = PromptTemplate.from_template(
        "ROLE: You are an expert Clinical Ontologist.\n"
        "TASK: Extract an interconnected clinical knowledge network regarding the diagnosis: {topic}.\n"
        "TEXT:\n{text}\n\n"
        "INSTRUCTION: Extract exact logical relationships (Triplets) linking {topic} to its Symptoms, Causes, Diagnostic Tests, and Treatments. "
        "Also link specific symptoms to treatments if applicable. DO NOT create disconnected floating nodes.\n"
        "FORMAT INSTRUCTIONS: {format_instructions}"
    )
    kg_chain = kg_prompt | llm | kg_parser
    kg_data = None
    try:
        winning_docs = [d for d in compressed_docs if d.metadata["topic"] == winning_topic]
        # Trim text significantly to save tokens and prevent rate limit exhaustion
        combined_text = "\n".join([d.page_content[:600] for d in winning_docs[:2]])
        kg_data_obj = kg_chain.invoke({
            "topic": winning_topic, 
            "text": combined_text,
            "format_instructions": kg_parser.get_format_instructions()
        })
        
        # Convert Pydantic object to dict
        kg_data = {
            "topic": kg_data_obj.topic,
            "Relationships": [{"subject": r.subject, "relation": r.relation, "object": r.object} for r in kg_data_obj.relationships]
        }
        
        # Try Neo4j write as best-effort (non-blocking)
        try:
            db_result = update_neo4j_graph(kg_data)
            if "Neo4j Error" in db_result:
                print(f"[!] {db_result}")
            elif "Skipped" in db_result:
                print(f"[*] {db_result}")
            else:
                print(f"[*] Neo4j Graph updated successfully for {winning_topic}!")
        except Exception as neo_err:
            print(f"[*] Neo4j write skipped: {neo_err}")
            
    except Exception as e:
        print(f"[*] KG extraction failed (graph will be empty): {e}")

    # 6. Build graph data LOCALLY from extracted KG (always works, no Neo4j dependency)
    if kg_data:
        graph_data = build_local_graph_data(kg_data)
        graph_logic_text = "\n".join(
            [f"- {r['subject']} {r['relation']} {r['object']}" for r in kg_data.get("Relationships", [])]
        )
        graph_logic = f"Knowledge Graph Context:\n{graph_logic_text}" if graph_logic_text else "No clinical logic extracted."
    else:
        graph_data = {"nodes": [], "links": []}
        graph_logic = "No clinical logic extracted."

    winning_docs = [d for d in compressed_docs if d.metadata["topic"] == winning_topic]
    retrieved_literature = "\n\n".join([d.page_content for d in winning_docs])

    # Extract abstracts and use LLM to translate them into clinical relevance bullet points
    abstracts = []
    
    relevance_prompt = PromptTemplate.from_template(
        "ROLE: You are a Clinical Research Assistant.\n"
        "TASK: Explain EXACTLY how the following medical paper relates to the patient's symptoms.\n"
        "PATIENT SYMPTOMS: {query}\n\n"
        "PAPER TITLE: {title}\n"
        "PAPER ABSTRACT: {abstract}\n\n"
        "RULES:\n"
        "- Keep the explanation extremely short, sharp, and bulleted (max 3 bullets).\n"
        "- Explain ONLY why this paper is relevant to the symptoms.\n"
        "- Do not use filler introductions like 'This paper is relevant because...'\n"
    )
    relevance_chain = relevance_prompt | llm | StrOutputParser()

    for doc in winning_docs[:3]: # Limit to top 3 to keep latency low
        content = doc.page_content
        lines = content.split('\n')
        title = ""
        abstract = ""
        for line in lines:
            if line.startswith("Title: "):
                title = line.replace("Title: ", "")
            elif line.startswith("Abstract: "):
                abstract = line.replace("Abstract: ", "")
        
        if title and abstract:
            try:
                relevance_explanation = relevance_chain.invoke({
                    "query": user_query,
                    "title": title,
                    "abstract": abstract[:1000] # Give the LLM enough context
                })
            except Exception:
                relevance_explanation = abstract[:500] + "..." # Fallback
                
            abstracts.append({
                "title": title, 
                "abstract": relevance_explanation,
                "url": doc.metadata.get("url", "")
            })

    # 7. Final Synthesis
    synthesis_prompt = PromptTemplate.from_template(
        template = """ROLE: You are Curo AI, an empathetic Clinical AI Assistant. You are a STRICT data interpreter. You do not invent medical logic. You only explain the data provided to you.

USER SYMPTOMS: {query}
PATIENT DATA (Vitals/Demo): {demography}
LOCAL PATIENT DOCUMENTS (PDF): {local_context}
MEDICAL LITERATURE (Chroma DB): {literature}
CLINICAL GUIDELINES (Real-time Scraping): {pathways}
CLINICAL LOGIC (Neo4j Graph): {graph}

INSTRUCTIONS & STRICT CONSTRAINTS:
1. EMPATHY: Acknowledge the user's symptoms warmly.
2. DATA PRIORITIZATION: Prioritize findings in the 'LOCAL PATIENT DOCUMENTS' if they are provided. This text comes directly from the patient's medical history/records.
3. TREATMENT PATHWAYS: Use the 'CLINICAL GUIDELINES' section to provide evidence-based treatment suggestions and diagnostic pathways.
4. GROUNDED SYNTHESIS: Synthesize literature, graph, and local context correctly.
5. CITATION: Weave findings naturally, explicitly mentioning 'your records' when referencing LOCAL PATIENT DOCUMENTS.
6. NEXT STEPS: Provide safe, non-diagnostic next steps.
7. DISCLAIMER: End with standard medical disclaimer.""",
        validate_template=True
    )
    
    synthesis_chain = synthesis_prompt | llm | StrOutputParser()
    final_response = synthesis_chain.invoke({
        "query": user_query,
        "demography": str(demography) if demography else "None provided",
        "local_context": local_context if local_context else "No local documents uploaded.",
        "literature": retrieved_literature,
        "pathways": clinical_pathways,
        "graph": graph_logic
    })

    vectorstore.delete_collection()

    return {
        "response": final_response,
        "extracted_ddx": clinical_terms,
        "winning_diagnosis": winning_topic,
        "abstracts": abstracts,
        "graph_data": graph_data
    }


def process_chat_followup(message: str, history: list, context: str, user_id: str = None) -> str:
    """Handles medical follow-up questions using the previous analysis context and stored health records."""
    
    # Auto-retrieve health records context if user_id is provided
    records_context = None
    if user_id:
        try:
            records_context = get_health_records_context(message, user_id)
        except Exception:
            pass
    
    chat_prompt = [
        SystemMessage(content=(
            "You are Curo AI, a clinical assistant. You are helping a user with follow-up questions "
            "based on a previous clinical analysis you performed.\n\n"
            f"ORIGINAL CLINICAL ANALYSIS CONTEXT:\n{context}\n\n"
            f"PATIENT HEALTH RECORDS:\n{records_context if records_context else 'None stored'}\n\n"
            "STRICT RULES:\n"
            "1. Stick to the context of the original analysis and any provided patient health records.\n"
            "2. If asked about patient history, prioritize information from PATIENT HEALTH RECORDS.\n"
            "3. Remain empathetic but data-driven.\n"
            "4. Do not make new diagnostic claims not supported by the medical literature or patient records.\n"
            "5. If a question suggests a worsening of condition, advise immediate medical consultation."
        ))
    ]
    
    # Add history
    for msg in history:
        if msg['role'] == 'user':
            chat_prompt.append(HumanMessage(content=msg['content']))
        else:
            chat_prompt.append(AIMessage(content=msg['content']))
            
    # Add current message
    chat_prompt.append(HumanMessage(content=message))
    
    response = llm.invoke(chat_prompt)
    return response.content

def clean_json_response(content: str) -> str:
    """Strips markdown and extra text to find the raw JSON object."""
    import re
    # Look for content between ```json and ``` or just ``` and ```
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: Find the first '{' and last '}'
    start = content.find('{')
    end = content.rfind('}')
    if start != -1 and end != -1:
        return content[start:end+1].strip()
    
    return content.strip()

def process_triage_chat(history: list, count: int) -> dict:
    """Uses the LLM to dynamically generate the next triage question and options using Pydantic validation.
    Now handles count==0 to generate the opening question (fully agent-driven, no hardcoded tree)."""
    
    # 1. Check if we should conclude
    if count >= 5:
        conclusion_prompt = [
            SystemMessage(content=(
                "You are an Expert Clinical Triage Assistant. Based on the following interview history, "
                "synthesize a comprehensive professional clinical summary of the patient's presentation. "
                "STRICT RULE: Do not include or assume any demographic information (age, sex, etc.) unless it was explicitly provided in the history.\n\n"
                f"{triage_parser.get_format_instructions()}"
            )),
            HumanMessage(content=f"INTERVIEW HISTORY:\n{str(history)}")
        ]
        try:
            response = triage_llm.invoke(conclusion_prompt)
            cleaned_content = clean_json_response(response.content)
            res = triage_parser.parse(cleaned_content)
            return res.model_dump()
        except Exception as e:
            print(f"[*] Triage Conclusion Parsing Error: {e}")
            return {"finished": True, "summary": "Patient reports varying symptoms requiring clinical investigation."}

    # 2. Generate the next question (or opening question if count == 0)
    if count == 0:
        # Opening question — fully LLM-driven, no hardcoded tree
        system_content = (
            "You are a Senior Clinical Triage Officer at Curo AI. You are starting a new diagnostic interview with a patient.\n"
            "Generate an opening question to begin the clinical intake. Ask where the primary clinical concern is focused.\n\n"
            "RULES:\n"
            "1. Provide 5-6 broad clinical category options (e.g., Head & Neurological, Chest & Respiratory, Abdomen & Digestive, etc.)\n"
            "2. ALWAYS include 'Other (please specify)' as the final option.\n"
            "3. Be warm and professional.\n"
            "4. FORMAT: Return a valid JSON object.\n"
            f"5. {triage_parser.get_format_instructions()}"
        )
        triage_prompt = [
            SystemMessage(content=system_content),
            HumanMessage(content="Begin the clinical interview.")
        ]
    else:
        # Follow-up questions
        triage_prompt = [
            SystemMessage(content=(
                "You are a friendly and warm Healthcare Assistant at Curo AI. You are having a casual but thorough conversation with a patient.\n"
                "Your goal is to understand their symptoms in simple, clear terms to help guide them toward the right care.\n\n"
                "IMPORTANT RULES:\n"
                "1. KEEP IT SIMPLE: Ask questions using everyday language that any patient can understand. Avoid complex medical jargon.\n"
                "2. NO REDUNDANCY: Never re-ask something already covered in the conversation history.\n"
                "3. BE WARM: Use a caring, conversational tone. Make the patient feel heard.\n"
                "4. GENERIC OPTIONS: Provide 3-4 simple, clear answer choices that cover the most common possibilities.\n"
                "5. ALWAYS INCLUDE 'Other (please specify)' as the LAST option in EVERY question — patients must always have the freedom to describe their experience in their own words.\n"
                "6. ASK ABOUT: duration, severity, what makes it better/worse, associated symptoms, and any medications.\n"
                "7. FORMAT: You MUST return a valid JSON object.\n"
                f"8. {triage_parser.get_format_instructions()}\n\n"
                "CONVERSATION SO FAR:"
            ))
        ]
    
        for msg in history:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'user':
                triage_prompt.append(HumanMessage(content=str(content)))
            else:
                if isinstance(content, dict):
                    question = content.get('question', str(content))
                else:
                    question = str(content)
                triage_prompt.append(AIMessage(content=question))

    response = triage_llm.invoke(triage_prompt)
    try:
        cleaned_content = clean_json_response(response.content)
        res = triage_parser.parse(cleaned_content)
        return res.model_dump()
    except Exception as e:
        print(f"[*] Triage Pydantic Parsing Error: {e} | Raw: {response.content[:150]}")
        if count == 0:
            return {
                "question": "Welcome! Where is your primary clinical concern focused?",
                "options": ["Head & Neurological", "Chest & Respiratory", "Abdomen & Digestive", "Bones & Joints", "Skin & Appearance", "Mood & Mental State", "Other (please specify)"],
                "finished": False
            }
        return {
            "question": "Can you provide more details about any other symptoms or how long this has been happening?",
            "options": ["Just started (last 24h)", "Few days", "Over a week", "Other (please specify)"],
            "finished": False
        }

def process_asr(audio_bytes: bytes) -> str:
    """Transcribes audio from memory-buffer using standard Scipy/Numpy."""
    try:
        # Load WAV from bytes
        samplerate, data = wavfile.read(io.BytesIO(audio_bytes))
        
        # Handle stereo → mono
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        
        # Convert to float32 and normalize
        if data.dtype != np.float32:
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            else:
                data = data.astype(np.float32)
            
        # Whisper expects 16kHz
        if samplerate != 16000:
            from scipy.signal import resample
            num_samples = int(len(data) * 16000 / samplerate)
            data = resample(data, num_samples)
            samplerate = 16000
        
        # Skip if audio is too short (less than 0.5 seconds)
        if len(data) < samplerate * 0.5:
            return "Recording too short. Please try again."
            
        pipe = get_asr_pipe()
        result = pipe(data, generate_kwargs={"return_timestamps": True})
        transcript = result["text"].strip()
        
        if not transcript:
            return "No speech detected. Please try again."
            
        return transcript
    except Exception as e:
        print(f"[*] ASR Error: {e}")
        import traceback
        traceback.print_exc()
        return f"Transcription error: {str(e)}"


# ============================================================================
# HEALTH RECORDS VAULT — Persistent Multi-PDF RAG System
# ============================================================================

# Persistent ChromaDB storage directory
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")
Path(CHROMA_PERSIST_DIR).mkdir(exist_ok=True)

# Lazy-loaded EasyOCR reader (heavy import, only load when needed)
_ocr_reader = None

def _get_ocr_reader():
    """Lazy-load EasyOCR reader to avoid startup overhead."""
    global _ocr_reader
    if _ocr_reader is None:
        print("[*] Loading EasyOCR reader for image-based PDF extraction...")
        import easyocr
        _ocr_reader = easyocr.Reader(['en'], gpu=False)
    return _ocr_reader


def extract_text_from_pdf(pdf_bytes: bytes, filename: str = "document.pdf") -> str:
    """
    Robust PDF text extraction with multiple backends:
    1. PyMuPDF (fitz) — best quality, supports OCR fallback for scanned pages
    2. pypdf — fallback if PyMuPDF not installed
    """
    # Try PyMuPDF first (best quality)
    try:
        try:
            import fitz
        except ImportError:
            import pymupdf as fitz
        
        return _extract_with_pymupdf(fitz, pdf_bytes, filename)
    except ImportError:
        print(f"[*] PyMuPDF not available, falling back to pypdf for {filename}")
        return _extract_with_pypdf(pdf_bytes, filename)


def _extract_with_pymupdf(fitz, pdf_bytes: bytes, filename: str) -> str:
    """Extract text using PyMuPDF with OCR fallback for image pages."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Try standard text extraction first
        text = page.get_text("text").strip()
        
        if len(text) > 50:
            full_text.append(f"--- Page {page_num + 1} ---\n{text}")
        else:
            # Page is likely image-based — use OCR
            try:
                mat = fitz.Matrix(300 / 72, 300 / 72)
                pix = page.get_pixmap(matrix=mat)
                img_bytes = pix.tobytes("png")
                
                from PIL import Image
                img = Image.open(io.BytesIO(img_bytes))
                img_np = np.array(img)
                
                reader = _get_ocr_reader()
                results = reader.readtext(img_np, detail=0, paragraph=True)
                ocr_text = "\n".join(results).strip()
                
                if ocr_text:
                    full_text.append(f"--- Page {page_num + 1} (OCR) ---\n{ocr_text}")
                elif text:
                    full_text.append(f"--- Page {page_num + 1} ---\n{text}")
                    
            except Exception as e:
                print(f"[!] OCR failed for page {page_num + 1} of {filename}: {e}")
                if text:
                    full_text.append(f"--- Page {page_num + 1} ---\n{text}")
    
    doc.close()
    return "\n\n".join(full_text)


def _extract_with_pypdf(pdf_bytes: bytes, filename: str) -> str:
    """Fallback extractor using pypdf (no OCR support, but handles text PDFs)."""
    from pypdf import PdfReader
    
    pdf = PdfReader(io.BytesIO(pdf_bytes))
    full_text = []
    
    for page_num, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text and text.strip():
            full_text.append(f"--- Page {page_num + 1} ---\n{text.strip()}")
    
    return "\n\n".join(full_text)


def _get_user_collection_name(user_id: str) -> str:
    """Generate a ChromaDB-safe collection name from user_id."""
    safe_id = hashlib.md5(user_id.encode()).hexdigest()[:16]
    return f"hr_{safe_id}"


def process_health_records(file_data: list, user_id: str) -> dict:
    """
    Processes multiple uploaded PDFs: extracts text, chunks, embeds, and stores
    in a persistent per-user ChromaDB collection.
    """
    collection_name = _get_user_collection_name(user_id)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    all_documents = []
    processing_results = []
    
    for file_info in file_data:
        filename = file_info["filename"]
        content = file_info["content"]
        
        try:
            extracted_text = extract_text_from_pdf(content, filename)
            
            if not extracted_text.strip():
                processing_results.append({
                    "filename": filename,
                    "status": "error",
                    "message": "No text could be extracted from this PDF."
                })
                continue
            
            chunks = text_splitter.split_text(extracted_text)
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "filename": filename,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "user_id": user_id,
                        "source_type": "health_record"
                    }
                )
                all_documents.append(doc)
            
            processing_results.append({
                "filename": filename,
                "status": "success",
                "chunks": len(chunks),
                "characters": len(extracted_text)
            })
            
            print(f"[*] Processed {filename}: {len(chunks)} chunks, {len(extracted_text)} chars")
            
        except Exception as e:
            print(f"[!] Failed to process {filename}: {e}")
            processing_results.append({
                "filename": filename,
                "status": "error",
                "message": str(e)
            })
    
    if not all_documents:
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "results": processing_results
        }
    
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        
        vectorstore.add_documents(all_documents)
        print(f"[*] Stored {len(all_documents)} chunks in collection '{collection_name}'")
        
    except Exception as e:
        print(f"[!] ChromaDB storage error: {e}")
        return {
            "total_documents": len(file_data),
            "total_chunks": 0,
            "results": processing_results,
            "error": f"Storage failed: {str(e)}"
        }
    
    return {
        "total_documents": len(file_data),
        "total_chunks": len(all_documents),
        "results": processing_results
    }


def query_health_records(query: str, user_id: str) -> dict:
    """
    RAG query against a user's stored health records.
    Retrieves relevant chunks and synthesizes an answer with source citations.
    """
    collection_name = _get_user_collection_name(user_id)
    
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        
        collection_data = vectorstore.get()
        if not collection_data or not collection_data.get("ids"):
            return {
                "answer": "You don't have any health records uploaded yet. Please upload your medical documents first.",
                "sources": []
            }
        
    except Exception as e:
        return {
            "answer": "Unable to access your health records. Please try again or re-upload your documents.",
            "sources": [],
            "error": str(e)
        }
    
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        relevant_docs = retriever.invoke(query)
    except Exception as e:
        return {
            "answer": f"Search failed: {str(e)}",
            "sources": []
        }
    
    if not relevant_docs:
        return {
            "answer": "I couldn't find relevant information in your health records for this query. Try rephrasing or ask about specific test results, medications, or diagnoses.",
            "sources": []
        }
    
    context_parts = []
    sources = []
    seen_chunks = set()
    
    for doc in relevant_docs:
        chunk_key = doc.page_content[:100]
        if chunk_key in seen_chunks:
            continue
        seen_chunks.add(chunk_key)
        
        context_parts.append(doc.page_content)
        sources.append({
            "filename": doc.metadata.get("filename", "Unknown"),
            "chunk_text": doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""),
            "chunk_index": doc.metadata.get("chunk_index", 0)
        })
    
    retrieved_context = "\n\n".join(context_parts)
    
    synthesis_prompt = PromptTemplate.from_template(
        "ROLE: You are Curo AI, a clinical health records analyst.\n"
        "TASK: Answer the patient's question based STRICTLY on their health records below.\n\n"
        "PATIENT'S HEALTH RECORDS:\n{context}\n\n"
        "PATIENT'S QUESTION: {query}\n\n"
        "RULES:\n"
        "1. ONLY use information present in the health records above. Do NOT invent data.\n"
        "2. If the records contain test results, quote the exact values.\n"
        "3. If the answer is not in the records, say 'This information was not found in your uploaded records.'\n"
        "4. Mention which document/report the information came from when possible.\n"
        "5. Be clear, concise, and clinically accurate.\n"
        "6. If you notice any abnormal values, flag them with appropriate context.\n"
        "7. End with a brief note if any results need professional follow-up."
    )
    
    synthesis_chain = synthesis_prompt | llm | StrOutputParser()
    
    try:
        answer = synthesis_chain.invoke({
            "context": retrieved_context,
            "query": query
        })
    except Exception as e:
        answer = f"Failed to synthesize answer: {str(e)}"
    
    return {
        "answer": answer,
        "sources": sources
    }


def get_health_records_context(query: str, user_id: str, k: int = 4) -> Optional[str]:
    """
    Lightweight retrieval-only function for the main analysis pipeline.
    Returns relevant health record chunks as concatenated text, or None.
    """
    collection_name = _get_user_collection_name(user_id)
    
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        
        collection_data = vectorstore.get()
        if not collection_data or not collection_data.get("ids"):
            return None
        
    except Exception:
        return None
    
    try:
        docs = vectorstore.similarity_search(query, k=k)
        if not docs:
            return None
        
        context_parts = []
        for doc in docs:
            filename = doc.metadata.get("filename", "Unknown")
            context_parts.append(f"[Source: {filename}]\n{doc.page_content}")
        
        return "\n\n".join(context_parts)
        
    except Exception:
        return None


def list_health_records(user_id: str) -> dict:
    """Returns metadata about documents stored in the user's health records collection."""
    collection_name = _get_user_collection_name(user_id)
    
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        
        collection_data = vectorstore.get()
        
        if not collection_data or not collection_data.get("ids"):
            return {"documents": [], "total_chunks": 0}
        
        file_stats = {}
        for metadata in collection_data.get("metadatas", []):
            fname = metadata.get("filename", "Unknown")
            if fname not in file_stats:
                file_stats[fname] = {"filename": fname, "chunks": 0}
            file_stats[fname]["chunks"] += 1
        
        return {
            "documents": list(file_stats.values()),
            "total_chunks": len(collection_data["ids"])
        }
        
    except Exception as e:
        return {"documents": [], "total_chunks": 0, "error": str(e)}


def delete_health_records(user_id: str, filename: str = None) -> dict:
    """
    Deletes health records from the user's collection.
    If filename is provided, deletes only that file's chunks.
    If filename is None, deletes the entire collection.
    """
    collection_name = _get_user_collection_name(user_id)
    
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        
        if filename:
            collection_data = vectorstore.get()
            ids_to_delete = []
            
            for i, metadata in enumerate(collection_data.get("metadatas", [])):
                if metadata.get("filename") == filename:
                    ids_to_delete.append(collection_data["ids"][i])
            
            if ids_to_delete:
                vectorstore.delete(ids=ids_to_delete)
                return {"deleted": len(ids_to_delete), "filename": filename}
            else:
                return {"deleted": 0, "message": f"No records found for file: {filename}"}
        else:
            vectorstore.delete_collection()
            return {"deleted": "all", "message": "All health records cleared."}
            
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# GENERIC MODE — Web Search ReAct Agent for Common Ailments
# ============================================================================

def _scrape_page_text(url: str, timeout: int = 8) -> str:
    """Fetches a webpage and extracts readable text content. Returns empty string on failure."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        
        html = resp.text
        
        # Remove script and style blocks
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<nav[^>]*>.*?</nav>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<footer[^>]*>.*?</footer>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<header[^>]*>.*?</header>', '', html, flags=re.DOTALL | re.IGNORECASE)
        
        # Strip all remaining tags
        text = re.sub(r'<[^>]+>', ' ', html)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Decode HTML entities
        import html as html_module
        text = html_module.unescape(text)
        
        # Return a meaningful chunk (first ~3000 chars of body text)
        return text[:3000] if len(text) > 3000 else text
        
    except Exception as e:
        print(f"[!] Scrape failed for {url[:60]}: {e}")
        return ""


def process_generic_query(user_query: str, user_id: str = None) -> dict:
    """
    Handles generic/simple medical queries using deep DuckDuckGo web search.
    Performs multi-angle research, scrapes actual page content from top results,
    and synthesizes a comprehensive, cited response.
    """
    from duckduckgo_search import DDGS
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    print(f"--- [GENERIC MODE] Deep Research Processing: {user_query} ---\n")

    sources = []
    seen_urls = set()
    raw_results = []
    
    # --- Phase 1: Multi-angle DuckDuckGo search ---
    search_queries = [
        f"{user_query} medical treatment guidelines",
        f"{user_query} causes symptoms diagnosis",
        f"{user_query} home remedies natural treatment",
        f"{user_query} when to see doctor red flags",
    ]
    
    try:
        with DDGS() as ddgs:
            for sq in search_queries:
                try:
                    results = list(ddgs.text(sq, max_results=5))
                    for r in results:
                        url = r.get('href', '')
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            raw_results.append(r)
                except Exception as inner_e:
                    print(f"[!] Sub-query failed: {sq[:50]} — {inner_e}")
                    continue
    except Exception as e:
        print(f"[!] DuckDuckGo search init failed: {e}")

    print(f"[*] Collected {len(raw_results)} unique search results across {len(search_queries)} queries.")

    # --- Phase 2: Scrape actual page content from top results ---
    enriched_context = ""
    source_index = 0
    
    for r in raw_results[:12]:  # Process top 12 unique results
        url = r.get('href', '')
        title = r.get('title', 'N/A')
        snippet = r.get('body', 'N/A')
        
        # Attempt to scrape full page text for deeper context
        page_text = _scrape_page_text(url)
        
        source_index += 1
        
        if page_text and len(page_text) > 200:
            # Use scraped content (richer than snippet)
            enriched_context += f"[Source {source_index}] {title}\nURL: {url}\nContent: {page_text[:2000]}\n\n"
            print(f"  [✓] Scraped [{source_index}]: {title[:60]}... ({len(page_text)} chars)")
        else:
            # Fall back to DuckDuckGo snippet
            enriched_context += f"[Source {source_index}] {title}\nURL: {url}\nSnippet: {snippet}\n\n"
            print(f"  [~] Snippet [{source_index}]: {title[:60]}...")
        
        sources.append({
            "index": source_index,
            "title": title,
            "url": url,
            "snippet": snippet[:200] + "..." if len(snippet) > 200 else snippet
        })

    print(f"[*] Built enriched context from {len(sources)} sources ({len(enriched_context)} chars total).")

    # --- Phase 3: Deep LLM synthesis with inline citations ---
    prompt = PromptTemplate.from_template(
        "You are Curo AI, an expert medical research assistant. A user has a health-related question.\n"
        "You have been provided with extensively researched web sources below.\n\n"
        "USER QUERY: {query}\n\n"
        "RESEARCHED WEB SOURCES:\n{context}\n\n"
        "INSTRUCTIONS:\n"
        "Provide a thorough, well-researched, and empathetic response. Structure your answer as follows:\n\n"
        "- **Understanding Your Concern**: Briefly acknowledge the issue and explain what it is.\n"
        "- **Possible Causes**: What commonly causes this condition or symptom.\n"
        "- **Recommended Treatments**: Evidence-based OTC medications with proper dosages, and clinical treatments if applicable.\n"
        "- **Home Remedies & Lifestyle**: Natural and home-based approaches supported by the sources.\n"
        "- **When to See a Doctor**: Specific red flags and warning signs that warrant immediate medical attention.\n\n"
        "CRITICAL RULES:\n"
        "1. CITE your sources inline using [1], [2], [3] etc. corresponding to the source numbers provided. "
        "Place citations at the end of the specific claim or fact they support.\n"
        "2. Be thorough and detailed — do NOT give a surface-level answer. Explain WHY treatments work.\n"
        "3. Include specific dosages, durations, and practical instructions where the sources provide them.\n"
        "4. Do NOT use markdown headers (##). Use **bold text** for section labels.\n"
        "5. Maintain a warm, empathetic, and professional tone throughout.\n"
        "6. End with a clear medical disclaimer that this is informational only and not a substitute for professional medical advice.\n"
        "7. Do NOT fabricate information. Only cite facts found in the provided sources.\n"
    )

    try:
        chain = prompt | llm | StrOutputParser()
        response_text = chain.invoke({"query": user_query, "context": enriched_context})

        print(f"[*] Deep generic research response generated successfully.")

        return {
            "response": response_text,
            "mode": "generic",
            "sources": sources
        }

    except Exception as err:
        print(f"[!] Generic agent LLM failed: {err}")
        return {
            "response": f"Sorry, I couldn't process your request right now. Error: {str(err)}",
            "mode": "generic",
            "sources": []
        }