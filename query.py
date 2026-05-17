# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings

# # ---------------- LOAD EMBEDDINGS ----------------
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2"
# )

# # ---------------- LOAD FAISS ----------------
# vectorstore = FAISS.load_local(
#     "faiss_index",
#     embeddings,
#     allow_dangerous_deserialization=True
# )

# print("✅ FAISS loaded successfully")

# # ---------------- ADVANCED RETRIEVAL ----------------
# def advanced_retrieval(query, top_k=10):

#     print("\n🔍 Query:", query)

#     try:

#         docs_with_scores = vectorstore.similarity_search_with_score(
#             query,
#             k=top_k
#         )

#         if not docs_with_scores:

#             print("❌ No results found")
#             return []

#         results = []

#         for doc, distance in docs_with_scores:

#             # Better confidence score
#             score = round(100 / (1 + distance), 2)

#             results.append((doc, score))

#         # Sort highest confidence first
#         results.sort(key=lambda x: x[1], reverse=True)

#         print(f"🎯 Top Score: {results[0][1]:.2f}")
#         print(f"✅ Returning {len(results)} best matches")

#         return results

#     except Exception as e:

#         print(f"❌ Retrieval Error: {str(e)}")
#         return []


# # ---------------- INTERACTIVE ----------------
# if __name__ == "__main__":

#     while True:

#         query = input("\nAsk question (type 'exit' to quit): ")

#         if query.lower() == "exit":
#             break

#         results = advanced_retrieval(query)

#         for i, (doc, score) in enumerate(results):

#             print(f"\n===== Result {i+1} | Score: {score:.2f} =====")

#             print(doc.page_content)

#             print("\nMetadata:")
#             print(doc.metadata)
import re

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# ---------------- LOAD EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# ---------------- LOAD FAISS ----------------
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("✅ FAISS loaded successfully")


# ---------------- ENTITY EXTRACTION ----------------
def extract_entities(query):

    query_upper = query.upper()

    building = None
    category = None

    # ---------------- BUILDING EXTRACTION ----------------
    building_patterns = [

        r'([A-Z])\s*BUILDING',
        r'BUILDING\s*([A-Z])',

        r'([A-Z])\s*TOWER',
        r'TOWER\s*([A-Z])',

        r'([A-Z])\s*BLOCK',
        r'BLOCK\s*([A-Z])'
    ]

    for pattern in building_patterns:

        match = re.search(pattern, query_upper)

        if match:

            building = match.group(1)

            break

    # ---------------- CATEGORY EXTRACTION ----------------
    possible_categories = [

        "EARTH WORK",
        "R.C.C",
        "RCC",
        "RMC",
        "SOILING",
        "EXCAVATION",
        "BACKFILLING",
        "PCC",
        "PLASTER",
        "PAINT",
        "TILES",
        "FLOORING",
        "ELECTRICAL",
        "PLUMBING",
        "STEEL",
        "CONCRETE",
        "FOUNDATION",
        "MASONRY",
        "WATERPROOFING",
        "DOORS",
        "WINDOWS",
        "GLAZING",
        "ELEVATION"
    ]

    for item in possible_categories:

        if item in query_upper:

            category = item

            break

    return building, category


# ---------------- ADVANCED RETRIEVAL ----------------
def advanced_retrieval(query, top_k=10):

    print("\n🔍 Query:", query)

    try:

        # ---------------- ENTITY EXTRACTION ----------------
        building, category = extract_entities(query)

        print(f"🏢 Building Filter: {building}")
        print(f"📂 Category Filter: {category}")

        # ---------------- VECTOR SEARCH ----------------
        docs_with_scores = vectorstore.similarity_search_with_score(
            query,
            k=30
        )

        if not docs_with_scores:

            print("❌ No results found")

            return []

        filtered_results = []

        # ---------------- FILTER RESULTS ----------------
        for doc, distance in docs_with_scores:

            meta = doc.metadata

            doc_building = str(
                meta.get("building", "")
            ).upper()

            doc_head = str(
                meta.get("head", "")
            ).upper()

            building_match = True
            category_match = True

            # ---------------- BUILDING FILTER ----------------
            if building:

                if building not in doc_building:

                    building_match = False

            # ---------------- CATEGORY FILTER ----------------
            if category:

                clean_query_category = category.replace(".", "")
                clean_doc_head = doc_head.replace(".", "")

                if clean_query_category not in clean_doc_head:

                    category_match = False

            # ---------------- KEEP ONLY MATCHING DOCS ----------------
            if building_match and category_match:

                # Better confidence scoring
                score = round(
                    100 / (1 + distance),
                    2
                )

                filtered_results.append((doc, score))

        # ---------------- FALLBACK ----------------
        if not filtered_results:

            print("⚠️ No strict matches found. Using semantic fallback...")

            for doc, distance in docs_with_scores[:top_k]:

                score = round(
                    100 / (1 + distance),
                    2
                )

                filtered_results.append((doc, score))

        # ---------------- SORT ----------------
        filtered_results.sort(
            key=lambda x: x[1],
            reverse=True
        )

        # ---------------- TOP SCORE ----------------
        best_score = filtered_results[0][1]

        print(f"🎯 Top Score: {best_score:.2f}")
        print(f"✅ Returning {len(filtered_results[:top_k])} best matches")

        return filtered_results[:top_k]

    except Exception as e:

        print(f"❌ Retrieval Error: {str(e)}")

        return []


# ---------------- INTERACTIVE TEST ----------------
if __name__ == "__main__":

    while True:

        query = input("\nAsk question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        results = advanced_retrieval(query)

        for i, (doc, score) in enumerate(results):

            print(f"\n===== Result {i+1} | Score: {score:.2f} =====")

            print(doc.page_content)

            print("\nMetadata:")
            print(doc.metadata)




