from qdrant_client import QdrantClient

OUTPUT_FILE = "code_chunks.txt"
client = QdrantClient(url="http://localhost:6333")

collections = client.get_collections().collections
collection_names = [c.name for c in collections]
print(f"Found {len(collection_names)} collections: {collection_names}")

total_count = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for collection_name in collection_names:
        info = client.get_collection(collection_name)
        print(f"\n--- Collection: {collection_name} | Points: {info.points_count} ---")

        offset = None
        coll_count = 0

        f.write(f"\n{'='*60}\n")
        f.write(f"COLLECTION: {collection_name}\n")
        f.write(f"{'='*60}\n\n")

        while True:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            if not points:
                break

            for p in points:
                description = p.payload.get("description", "")
                text        = p.payload.get("text", "")

                # Reconstruct embed_text exactly as describer.py built it
                embed_text = f"{description}\n\n{text}".strip() if description else text

                coll_count += 1
                total_count += 1

                f.write(f"# CHUNK {coll_count} | ID: {p.id}\n\n")
                f.write(embed_text if embed_text else "[NO CONTENT]")
                f.write("\n\n" + "-" * 60 + "\n\n")

            print(f"  Scrolled {coll_count} chunks so far...")

            if next_offset is None:
                break

            offset = next_offset

        print(f"  Done: {coll_count} chunks from '{collection_name}'")

print(f"\n✅ Done. {total_count} total chunks across {len(collection_names)} collections saved to '{OUTPUT_FILE}'")