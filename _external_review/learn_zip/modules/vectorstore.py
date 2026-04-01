class SimpleVectorStore:
    def __init__(self):
        self.documents = []

    def add_document(self, text):
        if text and text.strip():
            self.documents.append(text.strip())

    def search(self, query, limit=3):
        if not query.strip():
            return []

        query_tokens = set(query.lower().split())
        scored = []
        for document in self.documents:
            document_tokens = set(document.lower().split())
            overlap = len(query_tokens.intersection(document_tokens))
            if overlap > 0:
                scored.append((overlap, document))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in scored[:limit]]
