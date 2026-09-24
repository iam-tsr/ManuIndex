class ONNXEmbedder:
    """A class to embed queries and documents using ONNX models."""

    def embed_query(self, query: str) -> list:
        """
        Dummy method to embed a query.
        """

        raise NotImplementedError("This method should be implemented to call the actual embedding service.")

    def embed_documents(self, documents: list) -> list:
        """
        Dummy method to embed a list of documents.
        """

        raise NotImplementedError("This method should be implemented to call the actual embedding service.")