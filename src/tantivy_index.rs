use crate::error::{Result, VectorDbError};
use parking_lot::RwLock;
use std::path::Path;
use std::sync::Arc;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument};

/// Tantivy full-text search index
pub struct TantivyIndex {
    index: Index,
    reader: IndexReader,
    writer: Arc<RwLock<IndexWriter>>,
    #[allow(dead_code)]
    schema: Schema,
    // Field handles
    id_field: Field,
    text_field: Field,
    metadata_field: Field,
}

impl TantivyIndex {
    /// Create a new Tantivy index
    pub fn new<P: AsRef<Path>>(index_path: P) -> Result<Self> {
        // Build schema
        let mut schema_builder = Schema::builder();

        // ID field (stored, indexed)
        let id_field = schema_builder.add_text_field("id", TEXT | STORED);

        // Full-text searchable content field
        let text_field = schema_builder.add_text_field("text", TEXT | STORED);

        // Metadata as JSON (stored, searchable)
        let metadata_field = schema_builder.add_json_field("metadata", STORED | TEXT);

        let schema = schema_builder.build();

        // Create index directory
        let index_path = index_path.as_ref();
        std::fs::create_dir_all(index_path).map_err(VectorDbError::Io)?;

        // Open or create index
        let index = Index::create_in_dir(index_path, schema.clone())
            .or_else(|_| Index::open_in_dir(index_path))
            .map_err(|e| {
                VectorDbError::InvalidParameter(format!("Failed to create Tantivy index: {}", e))
            })?;

        // Create writer (50MB heap)
        let writer = index.writer(50_000_000).map_err(|e| {
            VectorDbError::InvalidParameter(format!("Failed to create writer: {}", e))
        })?;

        // Create reader with auto-reload
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e| {
                VectorDbError::InvalidParameter(format!("Failed to create reader: {}", e))
            })?;

        Ok(Self {
            index,
            reader,
            writer: Arc::new(RwLock::new(writer)),
            schema,
            id_field,
            text_field,
            metadata_field,
        })
    }

    /// Add a document to the index
    pub fn add_document(&self, id: &str, text: &str, metadata: &serde_json::Value) -> Result<()> {
        let writer = self.writer.write();

        let doc = doc!(
            self.id_field => id,
            self.text_field => text,
            self.metadata_field => metadata.clone()
        );

        writer.add_document(doc).map_err(|e| {
            VectorDbError::InvalidParameter(format!("Failed to add document: {}", e))
        })?;

        Ok(())
    }

    /// Add multiple documents in batch
    pub fn add_documents_batch(
        &self,
        documents: &[(String, String, serde_json::Value)],
    ) -> Result<()> {
        let writer = self.writer.write();

        for (id, text, metadata) in documents {
            let doc = doc!(
                self.id_field => id.as_str(),
                self.text_field => text.as_str(),
                self.metadata_field => metadata.clone()
            );

            writer.add_document(doc).map_err(|e| {
                VectorDbError::InvalidParameter(format!("Failed to add document: {}", e))
            })?;
        }

        Ok(())
    }

    /// Delete a document by ID
    pub fn delete_document(&self, id: &str) -> Result<()> {
        let writer = self.writer.write();

        // Create a term for the ID field
        let id_term = tantivy::Term::from_field_text(self.id_field, id);

        // Delete the document
        writer.delete_term(id_term);

        Ok(())
    }

    /// Commit changes
    pub fn commit(&self) -> Result<()> {
        let mut writer = self.writer.write();
        writer
            .commit()
            .map_err(|e| VectorDbError::InvalidParameter(format!("Failed to commit: {}", e)))?;

        // Force reader reload
        self.reader.reload().map_err(|e| {
            VectorDbError::InvalidParameter(format!("Failed to reload reader: {}", e))
        })?;

        Ok(())
    }

    /// Search the index
    pub fn search(&self, query_str: &str, limit: usize) -> Result<Vec<(String, f32)>> {
        let searcher = self.reader.searcher();

        // Create query parser for text field
        let query_parser =
            QueryParser::for_index(&self.index, vec![self.text_field, self.metadata_field]);

        let query = query_parser.parse_query(query_str).map_err(|e| {
            VectorDbError::InvalidParameter(format!("Failed to parse query: {}", e))
        })?;

        // Search
        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(limit))
            .map_err(|e| VectorDbError::InvalidParameter(format!("Search failed: {}", e)))?;

        // Extract results
        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let retrieved_doc: TantivyDocument = searcher.doc(doc_address).map_err(|e| {
                VectorDbError::InvalidParameter(format!("Failed to retrieve document: {}", e))
            })?;

            if let Some(id_value) = retrieved_doc.get_first(self.id_field) {
                if let Some(id) = id_value.as_str() {
                    results.push((id.to_string(), score));
                }
            }
        }

        Ok(results)
    }

    /// Search with metadata filter
    #[allow(dead_code)]
    pub fn search_with_filter(
        &self,
        query_str: &str,
        metadata_query: Option<&str>,
        limit: usize,
    ) -> Result<Vec<(String, f32)>> {
        let searcher = self.reader.searcher();

        // Build combined query
        let query = if let Some(meta_q) = metadata_query {
            // Search both text and metadata
            let combined_query = format!("{} AND metadata:{}", query_str, meta_q);
            let parser =
                QueryParser::for_index(&self.index, vec![self.text_field, self.metadata_field]);
            parser.parse_query(&combined_query).map_err(|e| {
                VectorDbError::InvalidParameter(format!("Failed to parse query: {}", e))
            })?
        } else {
            let parser = QueryParser::for_index(&self.index, vec![self.text_field]);
            parser.parse_query(query_str).map_err(|e| {
                VectorDbError::InvalidParameter(format!("Failed to parse query: {}", e))
            })?
        };

        // Search
        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(limit))
            .map_err(|e| VectorDbError::InvalidParameter(format!("Search failed: {}", e)))?;

        // Extract results
        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let retrieved_doc: TantivyDocument = searcher.doc(doc_address).map_err(|e| {
                VectorDbError::InvalidParameter(format!("Failed to retrieve document: {}", e))
            })?;

            if let Some(id_value) = retrieved_doc.get_first(self.id_field) {
                if let Some(id) = id_value.as_str() {
                    results.push((id.to_string(), score));
                }
            }
        }

        Ok(results)
    }

    /// Get document count
    #[allow(dead_code)]
    pub fn num_docs(&self) -> u64 {
        self.reader.searcher().num_docs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_tantivy_index() {
        let temp_dir = TempDir::new().unwrap();
        let index = TantivyIndex::new(temp_dir.path()).unwrap();

        // Add documents using batch method
        let documents = vec![
            (
                "doc1".to_string(),
                "machine learning and artificial intelligence".to_string(),
                serde_json::json!({"category": "AI"}),
            ),
            (
                "doc2".to_string(),
                "deep learning neural networks".to_string(),
                serde_json::json!({"category": "ML"}),
            ),
        ];

        index.add_documents_batch(&documents).unwrap();
        index.commit().unwrap();

        // Search
        let results = index.search("machine learning", 10).unwrap();
        assert!(
            !results.is_empty(),
            "Search should return results after commit"
        );
        assert_eq!(results[0].0, "doc1");
    }
}
