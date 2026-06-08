import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.graphs import Neo4jGraph
# from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import Neo4jVector
# from langchain_neo4j import remove_lucene_chars
# from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
# from sentence_transformers import CrossEncoder

load_dotenv()

def remove_lucene_chars(text: str) -> str:
    """Remove Lucene special characters"""
    special_chars = [
        "+",
        "-",
        "&",
        "|",
        "!",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "^",
        '"',
        "~",
        "*",
        "?",
        ":",
        "\\",
    ]
    for char in special_chars:
        if char in text:
            text = text.replace(char, " ")
    return text.strip()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY]):
    raise RuntimeError("Missing .env vars: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, OPENAI_API_KEY")

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class Entities(BaseModel):
    """Identifying information about entities."""

    names: list[str] = Field(
        ...,
        description="All medical entities mentioned in the text, including diseases, symptoms, treatments, drugs, body parts, medical tests, risk factors, and causes",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting medical entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
)
entity_chain = llm.with_structured_output(Entities)

vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-large"),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding",
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)
vector_retriever = vector_index.as_retriever(k=5)

def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    if not words:
        return ""
    full_text_query = " AND ".join([f"{word}~2" for word in words])
    print(f"Generated Query: {full_text_query}")
    return full_text_query.strip()


# Fulltext index query
def graph_retriever(question: str) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke(question)
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('fulltext_entity_id', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": entity},
        )
        result += "\n".join([el['output'] for el in response])
    return result

# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# def rerank_chunks(question: str, chunks: list[str], top_k: int = 5) -> list[str]:
#     if not chunks:
#         return []
#     pairs = [(question, chunk) for chunk in chunks]
#     scores = cross_encoder.predict(pairs)
#     scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
#     return [chunk for _, chunk in scored[:top_k]]


def full_retriever(question: str) -> str:
    graph_data = graph_retriever(question)
    vector_data = [el.page_content for el in vector_retriever.invoke(question)]
    # reranked_vector_data = rerank_chunks(question, vector_data, top_k=5)

    final_data = f"""Graph data:
        {
        graph_data
        }
        Vector data:
        {
        vector_data
        }
            """
    print("=== FULL RETRIEVER OUTPUT ===", final_data)
    return final_data

template = """You are a helpful medical assistant.
only use data below to answer the question. If you do not have enough information, say you do not have enough information.

context:
{context}


Question: {question}
Use natural language and be concise.
Answer:"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": full_retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

def answer_question(question: str) -> str:
    return chain.invoke(question)


