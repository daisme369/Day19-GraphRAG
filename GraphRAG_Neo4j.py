# %% [markdown]
# # 🌟 Building a GraphRAG Knowledge Graph with Neo4j
#
# **Author:** VinAI – Day 19: GraphRAG  
# **Objective:** Learn how to build a Knowledge Graph from extracted triples and query it for multi-hop reasoning.
#
# ## What is GraphRAG?
# Traditional RAG retrieves flat text chunks via vector similarity. **GraphRAG** enhances this by:
# 1. **Extracting entities & relationships** from documents (triples: `head → relation → tail`)
# 2. **Storing them in a graph database** (Neo4j) where connections are first-class citizens
# 3. **Traversing the graph** to find multi-hop reasoning paths that vector search alone cannot discover
#
# ### Pipeline Overview
# ```
# Documents → LLM Extraction → Triples (JSON) → Neo4j Graph → Cypher Queries → LLM Answer
# ```
#
# In this notebook, we use **pre-extracted triples** from `triples.json` (simulating the LLM extraction step)
# and focus on graph construction, deduplication, visualization, and retrieval.

# %% [markdown]
# ---
# ## Step 1: Environment Setup & Dependencies

# %%
# Install required packages (uncomment if needed)
# !pip install neo4j pyvis python-dotenv

# %%
import json
import os
from collections import defaultdict

from dotenv import load_dotenv
from neo4j import GraphDatabase
from pyvis.network import Network
from IPython.display import HTML, display

# Load environment variables from .env file
load_dotenv()

print("✅ All libraries imported successfully!")

# %% [markdown]
# ---
# ## Step 2: Neo4j Connection
#
# We create a reusable `Neo4jConnection` class that:
# - Connects using the official `neo4j` Python driver
# - Provides a `query()` helper with **parameterized queries** (prevents Cypher injection)
# - Properly closes the driver when done
#
# > 🔑 **Setup:** Create a free Neo4j Aura instance at https://neo4j.com/cloud/aura-free/
# > then put your credentials in the `.env` file.

# %%
class Neo4jConnection:
    """Manages connection to a Neo4j database instance."""

    def __init__(self, uri: str, user: str, password: str):
        # The driver handles connection pooling internally
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        # Verify connectivity on init so we fail fast
        self._driver.verify_connectivity()
        print(f"✅ Connected to Neo4j at {uri}")

    def close(self):
        """Always close the driver when done to release resources."""
        self._driver.close()
        print("🔌 Neo4j connection closed.")

    def query(self, cypher: str, parameters: dict = None):
        """
        Execute a Cypher query and return results as a list of dicts.

        WHY parameterized queries?
        - Prevents Cypher injection (same idea as SQL injection prevention)
        - Neo4j can cache and reuse query plans for better performance
        """
        with self._driver.session() as session:
            result = session.run(cypher, parameters or {})
            return [record.data() for record in result]

# %%
# --- Connection Configuration ---
# Replace these with your Neo4j Aura credentials in .env file:
#   NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io
#   NEO4J_USERNAME=neo4j
#   NEO4J_PASSWORD=your_password_here

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://your-instance.databases.neo4j.io")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password_here")

conn = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

# %% [markdown]
# ---
# ## Step 3: Entity & Relationship Extraction (Pre-extracted Triples)
#
# In a production GraphRAG pipeline, an LLM (e.g., GPT-4, Gemini) would read each document
# and output structured triples like:
# ```json
# {"head": "OpenAI Global, LLC", "relation": "FOUNDED_BY", "tail": "Sam Altman", "source": "OpenAI"}
# ```
#
# **We skip the LLM call** and load pre-extracted triples from `triples.json`.
#
# ### Key observations in our data:
# - **Duplicate entities exist:** e.g., `"Microsoft"` vs `"Microsoft Corporation"` — we'll handle this
# - **Relations are uppercase:** `FOUNDED_BY`, `DEVELOPED`, `HEADQUARTERED_IN` — these become edge types
# - **Source field:** tracks which Wikipedia article the triple came from

# %%
# Load the pre-extracted triples
with open("triples.json", "r", encoding="utf-8") as f:
    triples = json.load(f)

print(f"📊 Loaded {len(triples)} triples from triples.json")
print(f"\n--- Sample triples (first 5) ---")
for t in triples[:5]:
    print(f"  {t['head']}  --[{t['relation']}]-->  {t['tail']}")

# %%
# Analyze the data: what entity names and relation types do we have?
heads = set(t["head"] for t in triples)
tails = set(t["tail"] for t in triples)
all_entities = heads | tails
relations = set(t["relation"] for t in triples)
sources = set(t["source"] for t in triples)

print(f"📌 Unique head entities : {len(heads)}")
print(f"📌 Unique tail entities : {len(tails)}")
print(f"📌 Total unique entities: {len(all_entities)}")
print(f"📌 Unique relation types: {len(relations)}")
print(f"📌 Source documents     : {sources}")
print(f"\n--- Relation types ---")
for r in sorted(relations):
    print(f"  • {r}")

# %% [markdown]
# ---
# ## Step 4: Graph Construction & Deduplication (Crucial Step)
#
# ### Why is deduplication critical?
# When an LLM extracts triples from multiple documents, the **same entity** often appears
# with slightly different names:
# - `"Microsoft"` vs `"Microsoft Corporation"`
# - `"AI"` vs `"artificial intelligence"`
#
# Without deduplication, we'd create **separate nodes** for the same real-world entity,
# breaking the graph's connectivity and making traversal queries incomplete.
#
# ### How Cypher `MERGE` solves this
# ```cypher
# MERGE (n:Entity {name: $name})
# ```
# `MERGE` is an **upsert** operation:
# - If a node with `name = "Sam Altman"` exists → **reuse** it
# - If it doesn't exist → **create** it
#
# This ensures each unique entity name maps to exactly **one** node.
#
# ### Our deduplication strategy
# We also build a simple **alias map** to normalize known duplicates before ingestion.

# %%
# --- Step 4a: Build an alias map for known duplicates ---
# In production, you'd use an LLM or fuzzy matching for this.
# Here we manually map known aliases found in our data.

ENTITY_ALIASES = {
    "Microsoft": "Microsoft Corporation",
    "AI": "artificial intelligence",
    # Add more aliases as you discover them in your data
}

def normalize_entity(name: str) -> str:
    """Normalize entity name using the alias map."""
    return ENTITY_ALIASES.get(name, name)

# %%
# --- Step 4b: Clear existing data (for re-runs) ---
print("🗑️  Clearing existing graph data...")
conn.query("MATCH (n) DETACH DELETE n")
print("✅ Graph cleared.")

# %%
# --- Step 4c: Ingest triples into Neo4j ---

def ingest_triples(conn: Neo4jConnection, triples: list):
    """
    Ingest a list of triples into Neo4j.
    
    Strategy:
    1. Normalize entity names via alias map
    2. MERGE head node (upsert - prevents duplicates)
    3. MERGE tail node (upsert - prevents duplicates)  
    4. MERGE the relationship between them
    5. SET the source property to track provenance
    """
    
    # WHY batch by relation type?
    # Cypher doesn't allow parameterized relationship types in MERGE.
    # So we group triples by relation and build one query template per type.
    
    by_relation = defaultdict(list)
    for t in triples:
        by_relation[t["relation"]].append(t)
    
    total_ingested = 0
    
    for rel_type, group in by_relation.items():
        # Sanitize relation type: remove special chars, replace spaces with underscores
        safe_rel = rel_type.replace(" ", "_").replace("&", "AND").replace("-", "_")
        
        # Build the Cypher query with the relation type baked in
        # (relation types cannot be parameterized in MERGE)
        cypher = f"""
        UNWIND $batch AS triple
        MERGE (h:Entity {{name: triple.head}})
        MERGE (t:Entity {{name: triple.tail}})
        MERGE (h)-[r:{safe_rel}]->(t)
        SET r.source = triple.source
        """
        
        # Normalize entity names before sending
        batch = []
        for t in group:
            batch.append({
                "head": normalize_entity(t["head"]),
                "tail": normalize_entity(t["tail"]),
                "source": t["source"]
            })
        
        conn.query(cypher, {"batch": batch})
        total_ingested += len(batch)
        print(f"  ✅ Ingested {len(batch):>3} triples with relation [{safe_rel}]")
    
    print(f"\n🎉 Total: {total_ingested} triples ingested into Neo4j!")

ingest_triples(conn, triples)

# %%
# --- Step 4d: Verify the graph ---
node_count = conn.query("MATCH (n) RETURN count(n) AS count")[0]["count"]
rel_count = conn.query("MATCH ()-[r]->() RETURN count(r) AS count")[0]["count"]
print(f"📊 Graph contains: {node_count} nodes, {rel_count} relationships")

# %% [markdown]
# ---
# ## Step 5: Visualizing the Graph in the Notebook
#
# Visualization helps us:
# - **Debug** data quality issues (missing connections, orphan nodes)
# - **Understand** the overall structure and clusters
# - **Communicate** the knowledge graph to stakeholders
#
# We use **PyVis** which renders an interactive HTML graph directly in the notebook.

# %%
def visualize_graph(conn: Neo4jConnection, limit: int = 150):
    """
    Fetch nodes & relationships from Neo4j and render with PyVis.
    Nodes are color-coded by their source document.
    """
    
    # Fetch all relationships (with a limit to keep visualization manageable)
    results = conn.query("""
        MATCH (h)-[r]->(t)
        RETURN h.name AS head, type(r) AS relation, t.name AS tail, r.source AS source
        LIMIT $limit
    """, {"limit": limit})
    
    # Create a PyVis network
    net = Network(
        height="700px",
        width="100%",
        bgcolor="#1a1a2e",       # Dark background for contrast
        font_color="#e0e0e0",    # Light text
        notebook=True,
        cdn_resources="remote"   # Use CDN for JS resources
    )
    
    # Physics settings for better layout
    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=200,
        spring_strength=0.01
    )
    
    # Color palette for different source documents
    SOURCE_COLORS = {
        "OpenAI": "#10b981",
        "Google": "#3b82f6",
        "Microsoft": "#f59e0b",
        "Meta Platforms": "#8b5cf6",
        "Apple Inc.": "#ef4444",
        "Amazon (company)": "#f97316",
        "Tesla, Inc.": "#ec4899",
        "NVIDIA": "#06b6d4",
        "Samsung Electronics": "#6366f1",
        "Intel": "#14b8a6",
    }
    DEFAULT_COLOR = "#64748b"
    
    # Track which nodes we've added (avoid duplicates in visualization)
    added_nodes = set()
    # Track node sources for coloring
    node_sources = {}
    
    for row in results:
        head, tail = row["head"], row["tail"]
        source = row.get("source", "")
        
        # Remember source for coloring
        if head not in node_sources:
            node_sources[head] = source
        if tail not in node_sources:
            node_sources[tail] = source
        
        # Add head node
        if head not in added_nodes:
            color = SOURCE_COLORS.get(source, DEFAULT_COLOR)
            net.add_node(head, label=head, color=color, size=20,
                        title=f"Entity: {head}\nSource: {source}")
            added_nodes.add(head)
        
        # Add tail node
        if tail not in added_nodes:
            color = SOURCE_COLORS.get(source, DEFAULT_COLOR)
            net.add_node(tail, label=tail, color=color, size=15,
                        title=f"Entity: {tail}\nSource: {source}")
            added_nodes.add(tail)
        
        # Add edge
        net.add_edge(head, tail, label=row["relation"], 
                    title=row["relation"], color="#4a5568")
    
    # Save and display
    output_file = "graph_visualization.html"
    net.save_graph(output_file)
    print(f"📊 Graph visualization: {len(added_nodes)} nodes, {len(results)} edges")
    print(f"💾 Saved to {output_file}")
    
    # Display inline in Jupyter
    display(HTML(open(output_file, "r", encoding="utf-8").read()))

visualize_graph(conn, limit=200)

# %% [markdown]
# ---
# ## Step 6: Graph Retrieval — Query Answering via Graph Traversal
#
# ### Vector Search vs. Graph Traversal
#
# | Feature | Vector Search (Traditional RAG) | Graph Traversal (GraphRAG) |
# |---|---|---|
# | **How it works** | Find text chunks with similar embeddings | Follow edges between connected nodes |
# | **Strength** | Semantic similarity ("what sounds related?") | Structural reasoning ("what IS connected?") |
# | **Weakness** | Can't do multi-hop reasoning | Requires structured extraction |
# | **Example** | "Find docs about Sam Altman" | "Who founded the company that developed ChatGPT?" |
#
# Graph traversal excels at **multi-hop questions** where you need to chain facts together.

# %%
# --- Query 1: Direct neighbors ---
# "What did OpenAI develop?"
print("=" * 70)
print("🔍 Query 1: What did OpenAI develop?")
print("=" * 70)

results = conn.query("""
    MATCH (company:Entity {name: $name})-[:DEVELOPED]->(product)
    RETURN product.name AS product
    ORDER BY product.name
""", {"name": "OpenAI Global, LLC"})

for r in results:
    print(f"  • {r['product']}")

# %%
# --- Query 2: Multi-hop traversal (2 hops) ---
# "What products are connected to Elon Musk's companies?"
print("=" * 70)
print("🔍 Query 2: What did companies founded by Elon Musk develop?")
print("=" * 70)

results = conn.query("""
    MATCH (person:Entity {name: $person})<-[:FOUNDED_BY]-(company)-[:DEVELOPED]->(product)
    RETURN company.name AS company, collect(product.name) AS products
""", {"person": "Elon Musk"})

for r in results:
    print(f"\n  🏢 {r['company']}:")
    for p in r["products"]:
        print(f"     • {p}")

# %%
# --- Query 3: Find shortest path between two entities ---
# This is a KEY GraphRAG capability: discovering non-obvious connections
print("=" * 70)
print("🔍 Query 3: Shortest path between Sam Altman and Jensen Huang")
print("=" * 70)

results = conn.query("""
    MATCH path = shortestPath(
        (a:Entity {name: $start})-[*..6]-(b:Entity {name: $end})
    )
    RETURN [n IN nodes(path) | n.name] AS entities,
           [r IN relationships(path) | type(r)] AS relations,
           length(path) AS hops
""", {"start": "Sam Altman", "end": "Jensen Huang"})

if results:
    r = results[0]
    print(f"  📏 Path length: {r['hops']} hops")
    print(f"  🔗 Path:")
    entities = r["entities"]
    relations = r["relations"]
    for i in range(len(relations)):
        print(f"     {entities[i]}  --[{relations[i]}]-->  {entities[i+1]}")
else:
    print("  ❌ No path found!")

# %%
# --- Query 4: Find common connections (Graph Pattern Matching) ---
# "Which companies are headquartered in California?"
print("=" * 70)
print("🔍 Query 4: Companies headquartered in California locations")
print("=" * 70)

results = conn.query("""
    MATCH (company:Entity)-[:HEADQUARTERED_IN]->(location:Entity)
    WHERE location.name CONTAINS 'California' OR location.name CONTAINS 'San'
    RETURN company.name AS company, location.name AS location
    ORDER BY company.name
""")

for r in results:
    print(f"  🏢 {r['company']}  →  📍 {r['location']}")

# %%
# --- Query 5: Simulate GraphRAG context retrieval ---
# Given a user question, retrieve relevant subgraph context for an LLM
print("=" * 70)
print("🔍 Query 5: GraphRAG Context Retrieval")
print("   User Question: 'Tell me about Microsoft's acquisitions and investments'")
print("=" * 70)

results = conn.query("""
    MATCH (ms:Entity {name: $company})-[r]->(target:Entity)
    WHERE type(r) IN ['ACQUIRED', 'INVESTED_BY', 'PROVIDES']
    RETURN type(r) AS relation, target.name AS target, r.source AS source
    ORDER BY type(r), target.name
""", {"company": "Microsoft Corporation"})

# Format as context string that would be sent to an LLM
context_lines = []
for r in results:
    line = f"Microsoft Corporation {r['relation']} {r['target']}"
    context_lines.append(line)
    print(f"  • {line}")

print(f"\n📝 Context retrieved: {len(context_lines)} facts")
print("   → This context would be injected into the LLM prompt for answering.")

# %% [markdown]
# ---
# ## Step 7: Graph Querying & Textualization (The GraphRAG Pipeline)
#
# A complete GraphRAG query goes through **4 sub-steps**:
#
# ```
# User Question
#     │
#     ▼
# ┌─────────────────────────┐
# │ 1. Entity Extraction    │  ← Identify key entities in the question
# └──────────┬──────────────┘
#            ▼
# ┌─────────────────────────┐
# │ 2. Graph Retrieval      │  ← Cypher: 2-hop traversal from those entities
# └──────────┬──────────────┘
#            ▼
# ┌─────────────────────────┐
# │ 3. Textualization       │  ← Convert graph paths into readable sentences
# └──────────┬──────────────┘
#            ▼
# ┌─────────────────────────┐
# │ 4. LLM Synthesis        │  ← Prompt = question + textualized context → answer
# └─────────────────────────┘
# ```
#
# ### Why "Textualization"?
# LLMs cannot read graph structures directly. We must convert the retrieved
# sub-graph into **natural-language sentences** so the LLM can reason over them.

# %%
# =============================================================================
# Sub-step 1: Entity Extraction (Mock)
# =============================================================================
# In production, you'd use an LLM or NER model to extract entities from the
# user question. Here we use a simple keyword-matching approach for education.

# Build an entity lookup from our graph: fetch all node names from Neo4j
all_entity_names = [
    r["name"] for r in conn.query("MATCH (n:Entity) RETURN n.name AS name")
]
print(f"📚 Loaded {len(all_entity_names)} entity names from the graph for matching.\n")

def extract_entities(question: str, entity_names: list) -> list:
    """
    Mock entity extraction: find which known entity names appear in the question.

    Strategy: sort entity names by length (longest first) so that
    'Microsoft Corporation' matches before 'Microsoft'.
    """
    found = []
    question_lower = question.lower()
    # Sort longest-first to prefer more specific matches
    for name in sorted(entity_names, key=len, reverse=True):
        if name.lower() in question_lower:
            found.append(name)
            # Remove matched text to avoid sub-matches
            # e.g., after matching "Microsoft Corporation", don't also match "Microsoft"
            question_lower = question_lower.replace(name.lower(), "")
    return found

# Quick test
test_q = "What products has Tesla, Inc. developed?"
print(f"Question: {test_q}")
print(f"Extracted: {extract_entities(test_q, all_entity_names)}")

# %%
# =============================================================================
# Sub-step 2: Graph Retrieval (2-hop traversal)
# =============================================================================

def retrieve_subgraph(conn: Neo4jConnection, entity_name: str, max_hops: int = 2) -> list:
    """
    Retrieve all paths up to `max_hops` from the given entity.

    Returns a list of dicts with keys: head, relation, tail.

    WHY 2 hops?
    - 1 hop  = direct facts ("Tesla developed Model 3")
    - 2 hops = chained facts ("Elon Musk → CEO_OF → Tesla → DEVELOPED → Model 3")
    - 3+ hops = usually too noisy, diminishing returns
    """
    cypher = """
        MATCH path = (start:Entity {name: $entity})-[*1..2]-(end)
        WITH relationships(path) AS rels
        UNWIND rels AS r
        WITH DISTINCT r
        RETURN startNode(r).name AS head, type(r) AS relation, endNode(r).name AS tail
    """
    return conn.query(cypher, {"entity": entity_name})

# Quick test
triples_result = retrieve_subgraph(conn, "Tesla, Inc.")
print(f"🔍 Retrieved {len(triples_result)} unique triples within 2 hops of 'Tesla, Inc.'")
for t in triples_result[:5]:
    print(f"   {t['head']}  --[{t['relation']}]-->  {t['tail']}")
print("   ...")

# %%
# =============================================================================
# Sub-step 3: Textualization
# =============================================================================

def textualize_triples(triples: list) -> str:
    """
    Convert a list of graph triples into readable English sentences.

    Example input:  {"head": "Tesla, Inc.", "relation": "DEVELOPED", "tail": "Model 3"}
    Example output: "Tesla, Inc. DEVELOPED Model 3."

    WHY this step?
    LLMs understand natural language, not raw graph structures.
    This bridges the gap between the graph DB and the language model.
    """
    if not triples:
        return "No relevant information found in the knowledge graph."

    sentences = []
    seen = set()  # Deduplicate identical sentences
    for t in triples:
        sentence = f"{t['head']} {t['relation'].replace('_', ' ')} {t['tail']}."
        if sentence not in seen:
            sentences.append(sentence)
            seen.add(sentence)

    return "\n".join(sentences)

# Quick test
context_text = textualize_triples(triples_result[:5])
print("--- Textualized context (first 5 triples) ---")
print(context_text)

# %%
# =============================================================================
# Sub-step 4: Full GraphRAG Pipeline
# =============================================================================

def query_graph_rag(question: str, conn: Neo4jConnection, entity_names: list) -> dict:
    """
    Complete GraphRAG query pipeline:
      Question → Entity Extraction → Graph Retrieval → Textualization → LLM Prompt

    Returns a dict with all intermediate results for inspection.
    """
    print(f"\n{'='*70}")
    print(f"❓ User Question: {question}")
    print(f"{'='*70}")

    # --- Step A: Extract entities ---
    entities = extract_entities(question, entity_names)
    print(f"\n📌 Step 1 — Extracted entities: {entities}")

    if not entities:
        print("⚠️  No entities found in question. Cannot query the graph.")
        return {"question": question, "entities": [], "triples": [],
                "context": "", "prompt": ""}

    # --- Step B: Retrieve sub-graph for each entity ---
    all_triples = []
    for entity in entities:
        sub_triples = retrieve_subgraph(conn, entity, max_hops=2)
        all_triples.extend(sub_triples)
        print(f"   🔗 '{entity}' → {len(sub_triples)} triples retrieved")

    # Deduplicate across entities
    unique_triples = []
    seen_keys = set()
    for t in all_triples:
        key = (t["head"], t["relation"], t["tail"])
        if key not in seen_keys:
            unique_triples.append(t)
            seen_keys.add(key)

    print(f"\n📊 Step 2 — Total unique triples: {len(unique_triples)}")

    # --- Step C: Textualize ---
    context = textualize_triples(unique_triples)
    print(f"\n📝 Step 3 — Textualized context ({len(context.splitlines())} sentences):")
    # Show first 8 lines to keep output manageable
    for line in context.splitlines()[:8]:
        print(f"   {line}")
    if len(context.splitlines()) > 8:
        print(f"   ... and {len(context.splitlines()) - 8} more sentences")

    # --- Step D: Construct the LLM prompt (mock) ---
    prompt = f"""You are a helpful AI assistant. Answer the user's question using ONLY
the provided context from the knowledge graph. If the context doesn't contain
enough information, say "I don't have enough information."

=== KNOWLEDGE GRAPH CONTEXT ===
{context}

=== USER QUESTION ===
{question}

=== YOUR ANSWER ==="""

    print(f"\n🤖 Step 4 — LLM Prompt constructed ({len(prompt)} chars)")
    print("   (In production, this prompt would be sent to GPT-4 / Gemini / etc.)")

    return {
        "question": question,
        "entities": entities,
        "triples": unique_triples,
        "context": context,
        "prompt": prompt,
    }

# %%
# --- Run the full pipeline on sample questions ---
result1 = query_graph_rag(
    "Who founded Tesla, Inc. and what products did they develop?",
    conn, all_entity_names
)

# %%
result2 = query_graph_rag(
    "What is the relationship between Elon Musk and OpenAI?",
    conn, all_entity_names
)

# %%
result3 = query_graph_rag(
    "Which companies has Microsoft Corporation acquired?",
    conn, all_entity_names
)

# %% [markdown]
# ---
# ## Step 8: Evaluation — GraphRAG vs. Flat RAG (The "Aha!" Moment)
#
# ### Why does Flat RAG struggle with multi-hop questions?
#
# **Flat RAG** (traditional vector-search RAG) works like this:
# 1. Embed the user question → vector
# 2. Find the top-K most **similar text chunks** from a corpus
# 3. Send those chunks + question to the LLM
#
# The problem: **each chunk is independent**. If the answer requires chaining
# facts from *different* chunks, the retriever often fails to find them all.
#
# **Example multi-hop question:**
# > *"Who is the CEO of the company that acquired Activision Blizzard?"*
#
# - Chunk A might say: *"Microsoft acquired Activision Blizzard"*
# - Chunk B might say: *"Bill Gates and Paul Allen founded Microsoft"*
# - Chunk C might say: *"Satya Nadella is the CEO of Microsoft"*
#
# Flat RAG might retrieve Chunk A (high similarity to "Activision Blizzard")
# but **miss Chunk C** (low similarity to "CEO"). The LLM then **hallucinates**.
#
# **GraphRAG** solves this by **traversing edges**:
# ```
# Activision Blizzard ←[ACQUIRED]— Microsoft Corporation —[CEO_OF]→ ???
# ```
# It follows the structural path and always retrieves the connected facts.
#
# ### Our Evaluation Strategy
# We compare **context retrieved** by each method on 5 multi-hop questions.

# %%
# =============================================================================
# Flat RAG Simulator (Mock)
# =============================================================================
# We simulate a flat RAG retriever using pre-built text chunks from our corpus.
# Each chunk is a paragraph about ONE company — mimicking what a real chunker
# would produce from Wikipedia articles.

FLAT_RAG_CHUNKS = {
    "openai_overview": (
        "OpenAI Global, LLC is an AI company founded in 2015 by Elon Musk, "
        "Sam Altman, Ilya Sutskever, Greg Brockman, and others. It is headquartered "
        "in San Francisco."
    ),
    "openai_products": (
        "OpenAI has developed the GPT family, DALL-E series, Sora series, and "
        "released ChatGPT in November 2022."
    ),
    "microsoft_overview": (
        "Microsoft Corporation was founded by Bill Gates and Paul Allen in 1975. "
        "It is headquartered in Redmond, Washington."
    ),
    "microsoft_acquisitions": (
        "Microsoft acquired Skype Technologies, LinkedIn, and Activision Blizzard. "
        "It provides Azure cloud computing platform."
    ),
    "tesla_overview": (
        "Tesla, Inc. was founded by Martin Eberhard and Marc Tarpenning in July 2003. "
        "Elon Musk is the CEO. It is headquartered in Austin, Texas."
    ),
    "tesla_products": (
        "Tesla developed the Roadster, Model S, Model X, Model 3, Model Y, "
        "Tesla Semi, and Cybertruck."
    ),
    "nvidia_overview": (
        "Nvidia Corporation was founded in 1993 by Jensen Huang, Chris Malachowsky, "
        "and Curtis Priem. It is headquartered in Santa Clara, California."
    ),
    "nvidia_products": (
        "Nvidia developed GPUs, SoCs, and APIs. It uses CUDA technology and "
        "controls 80% of the market for GPUs used in AI."
    ),
    "apple_overview": (
        "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. "
        "Tim Cook is the current CEO."
    ),
    "meta_overview": (
        "Meta Platforms, Inc. owns Facebook, Instagram, WhatsApp, Messenger, and Threads. "
        "Facebook was rebranded as Meta Platforms, Inc."
    ),
    "samsung_overview": (
        "Samsung Electronics Co., Ltd. was founded in 1969 and is headquartered in "
        "Suwon, South Korea. It is the largest vendor of smartphones."
    ),
    "amazon_overview": (
        "Amazon.com, Inc. was founded by Jeff Bezos in 1994. It developed "
        "Amazon Web Services (AWS) and acquired Whole Foods Market."
    ),
}

def flat_rag_retrieve(question: str, chunks: dict, top_k: int = 2) -> list:
    """
    Simulate flat RAG retrieval using simple keyword overlap scoring.

    In production, you'd use embeddings + cosine similarity (e.g., ChromaDB).
    Here we use word overlap as a transparent approximation so the intern
    can see exactly WHY certain chunks are retrieved or missed.
    """
    question_words = set(question.lower().split())

    scored = []
    for chunk_id, text in chunks.items():
        chunk_words = set(text.lower().split())
        # Jaccard-like overlap score
        overlap = len(question_words & chunk_words)
        scored.append((chunk_id, text, overlap))

    # Sort by overlap score descending, take top_k
    scored.sort(key=lambda x: x[2], reverse=True)
    return [(cid, text) for cid, text, _ in scored[:top_k]]

# Quick test
print("--- Flat RAG test ---")
test_results = flat_rag_retrieve("Who founded Tesla?", FLAT_RAG_CHUNKS)
for cid, text in test_results:
    print(f"  [{cid}]: {text[:80]}...")

# %%
# =============================================================================
# Define 5 Multi-Hop Evaluation Questions
# =============================================================================
# These questions REQUIRE chaining 2+ facts together.
# Flat RAG typically retrieves only part of the answer.

EVAL_QUESTIONS = [
    {
        "question": "Who is the CEO of the company that acquired Activision Blizzard?",
        "expected_chain": "Activision Blizzard ←[ACQUIRED]— Microsoft —[FOUNDED_BY]→ Bill Gates / Paul Allen",
        "expected_answer": "Microsoft Corporation acquired Activision Blizzard. Microsoft was founded by Bill Gates and Paul Allen.",
    },
    {
        "question": "What products were developed by the company that Elon Musk co-founded in 2015?",
        "expected_chain": "Elon Musk ←[FOUNDED_BY]— OpenAI —[DEVELOPED]→ GPT family, DALL-E, Sora",
        "expected_answer": "Elon Musk co-founded OpenAI in 2015. OpenAI developed the GPT family, DALL-E series, and Sora series.",
    },
    {
        "question": "Where is the headquarters of the company that developed the Cybertruck?",
        "expected_chain": "Cybertruck ←[DEVELOPED]— Tesla —[HEADQUARTERED_IN]→ Austin, Texas",
        "expected_answer": "Tesla, Inc. developed the Cybertruck. Tesla is headquartered in Austin, Texas.",
    },
    {
        "question": "What company invested in the organization that released ChatGPT?",
        "expected_chain": "ChatGPT ←[RELEASED]— OpenAI —[INVESTED_BY]→ Microsoft",
        "expected_answer": "OpenAI released ChatGPT. Microsoft invested in OpenAI.",
    },
    {
        "question": "Who founded the company that controls 80% of GPUs used in AI?",
        "expected_chain": "80% GPU market ←[CONTROLLED]— Nvidia —[FOUNDED_BY]→ Jensen Huang",
        "expected_answer": "Nvidia controls 80% of GPUs used in AI. Nvidia was founded by Jensen Huang, Chris Malachowsky, and Curtis Priem.",
    },
]

print(f"📋 Defined {len(EVAL_QUESTIONS)} multi-hop evaluation questions.")

# %%
# =============================================================================
# Run the Evaluation Loop
# =============================================================================

def run_evaluation(questions: list, conn: Neo4jConnection,
                   entity_names: list, chunks: dict):
    """
    For each question, compare Flat RAG vs GraphRAG retrieval.
    Prints a side-by-side report showing which method retrieves
    the complete chain of facts needed to answer correctly.
    """
    print("\n" + "=" * 80)
    print("📊  EVALUATION REPORT: GraphRAG vs. Flat RAG")
    print("=" * 80)

    graphrag_wins = 0
    flat_wins = 0
    ties = 0

    for i, q in enumerate(questions, 1):
        question = q["question"]

        print(f"\n{'─' * 80}")
        print(f"  Q{i}: {question}")
        print(f"  Expected chain: {q['expected_chain']}")
        print(f"{'─' * 80}")

        # --- Flat RAG ---
        flat_results = flat_rag_retrieve(question, chunks, top_k=2)
        flat_context = "\n".join([text for _, text in flat_results])
        flat_chunk_ids = [cid for cid, _ in flat_results]

        print(f"\n  📄 FLAT RAG (top-2 chunks): {flat_chunk_ids}")
        for cid, text in flat_results:
            # Truncate for readability
            preview = text[:100] + "..." if len(text) > 100 else text
            print(f"     [{cid}]: {preview}")

        # --- GraphRAG ---
        entities = extract_entities(question, entity_names)
        graph_triples = []
        for entity in entities:
            graph_triples.extend(retrieve_subgraph(conn, entity, max_hops=2))

        # Deduplicate
        seen = set()
        unique_graph = []
        for t in graph_triples:
            key = (t["head"], t["relation"], t["tail"])
            if key not in seen:
                unique_graph.append(t)
                seen.add(key)

        graph_context = textualize_triples(unique_graph)

        print(f"\n  🔗 GRAPH RAG (entities: {entities}, {len(unique_graph)} triples):")
        for line in graph_context.splitlines()[:6]:
            print(f"     {line}")
        if len(graph_context.splitlines()) > 6:
            print(f"     ... +{len(graph_context.splitlines()) - 6} more")

        # --- Verdict ---
        # Check if the expected answer keywords appear in each context
        answer_keywords = q["expected_answer"].lower().split()
        # Use a subset of important keywords for matching
        important_words = [w for w in answer_keywords
                          if len(w) > 4 and w not in {"the", "and", "was", "that", "from"}]

        flat_hits = sum(1 for w in important_words if w in flat_context.lower())
        graph_hits = sum(1 for w in important_words if w in graph_context.lower())
        flat_coverage = flat_hits / max(len(important_words), 1) * 100
        graph_coverage = graph_hits / max(len(important_words), 1) * 100

        if graph_coverage > flat_coverage:
            verdict = "🏆 GraphRAG WINS"
            graphrag_wins += 1
        elif flat_coverage > graph_coverage:
            verdict = "📄 Flat RAG wins"
            flat_wins += 1
        else:
            verdict = "🤝 Tie"
            ties += 1

        print(f"\n  📈 Coverage: Flat RAG = {flat_coverage:.0f}% | GraphRAG = {graph_coverage:.0f}%")
        print(f"  {verdict}")

    # --- Summary ---
    print(f"\n{'=' * 80}")
    print(f"📊  FINAL SCORE")
    print(f"{'=' * 80}")
    print(f"  🏆 GraphRAG wins : {graphrag_wins}/{len(questions)}")
    print(f"  📄 Flat RAG wins : {flat_wins}/{len(questions)}")
    print(f"  🤝 Ties          : {ties}/{len(questions)}")
    print()

    if graphrag_wins > flat_wins:
        print("  ✅ CONCLUSION: GraphRAG consistently outperforms Flat RAG")
        print("     on multi-hop reasoning questions because it retrieves")
        print("     structurally connected facts, not just similar text.")
    print(f"{'=' * 80}")

run_evaluation(EVAL_QUESTIONS, conn, all_entity_names, FLAT_RAG_CHUNKS)

# %% [markdown]
# ---
# ## 🧹 Cleanup

# %%
# Close the Neo4j connection
conn.close()
print("✅ Notebook complete! Full GraphRAG pipeline covered:")
print("   1. Loaded pre-extracted triples from documents")
print("   2. Ingested them into Neo4j with MERGE (deduplication)")
print("   3. Visualized the knowledge graph with PyVis")
print("   4. Performed multi-hop graph traversal queries")
print("   5. Simulated GraphRAG context retrieval for an LLM")
print("   6. Built a complete query pipeline (Extract → Retrieve → Textualize → Prompt)")
print("   7. Evaluated GraphRAG vs Flat RAG on multi-hop questions")
