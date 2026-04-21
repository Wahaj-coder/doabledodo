
# from qdrant_store import search, close_client
# from embedder import embed_texts

# REPO_URL   = "https://github.com/Wahaj-coder/abc___X"
# BRANCH     = "main"
# COLLECTION = "test_repo"

# q_vec = embed_texts(["cricket enocder lstm"])[0]

# results = search(
#     collection_name=COLLECTION,
#     query_vector=q_vec,
#     top_k=3,
#     filter_repo=REPO_URL,
#     filter_branch=BRANCH,
# )

# for r in results:
#     print(f"\n[{r['score']:.3f}] {r['file_path']}")
#     print(r['text'][:150])

# # close_client()
# # from chunker import chunk_repo

# # chunks = chunk_repo("./repos/abc___X")
# # for c in chunks:
# #     print(f"\n{'='*50}")
# #     print(f"FILE: {c['file_path']} | NAME: {c['name']}")
# #     print(c['text'])  # full text, no [:200] limit

from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

points, next_page = client.scroll(
    collection_name="abc1",
    limit=100,
    with_payload=True,
    with_vectors=False
)

for p in points:
    print("\nID:", p.id)
    print("Payload:", p.payload)