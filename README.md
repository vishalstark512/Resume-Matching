# Resume Matching AI pipeline

### Pre-processing steps:

1. **JSON to DataFrame**: The job description JSON file is converted into a pandas data frame to simplify data manipulation.
2. **Text Cleaning**: The code employs various natural language processing operations such as tokenization, stemming using WordNetLemmatizer, and removal of punctuations. A peculiar pattern of recurring "AAAA" is specifically cleaned from the text. There's a specific focus on retaining certain words, namely 'SAP', 'S4Hana', and 'ICT', and these are excluded from the stopwords list.
3. **Language Translation**: To maintain consistency, any resume in a non-English language is converted to English.
4. **Noise Reduction**: To ensure the text is meaningful and relevant, only words that are recognized in the English dictionary are retained, effectively filtering out any noise or non-sensical terms.

### Overall Strategy:

1. **Feature Extraction via Regex Patterns**: Before embedding the text, extract domain-specific features using predefined regex patterns.
2. **Embedding Texts**: Utilize a pre-trained SentenceTransformer model (`paraphrase-MiniLM-L6-v2`) to embed the textual content.
3. **Chunking Long Texts**: For texts that exceed the model's token limit, divide the text into overlapping chunks, embed each chunk separately, and then average their embeddings.
4. **Combining Embeddings**: Combine the embeddings obtained from the SentenceTransformer model with the domain-specific feature embeddings.
5. **Similarity Calculation**: Compute the cosine similarity between the job requirement's combined embedding and the combined embedding of each resume in the dataset.
6. **Ranking**: Rank the resumes based on their similarity scores to the job requirement.
7. **Output**: Save the detailed similarity scores and rankings to a CSV file and prepare a final submission CSV with the ID and rank.

### Cool Optimizations and Challenges Solved:

1. **Overlapping Chunking**:
    - *Challenge*: Direct chunking could miss some contextual information at the borders.
    - *Solution*: The implementation uses overlapping chunks to ensure that no information is lost at the boundaries between chunks. The average of the embeddings of these overlapping chunks provides a more holistic representation of the content.

2. **Domain-Specific Feature Embeddings**:
    - *Challenge*: Simple embeddings might not capture domain-specific nuances.
    - *Solution*: By defining a set of domain-specific features and using regex patterns to extract these features, the strategy combines traditional feature extraction with modern embedding techniques. This ensures that both general and domain-specific information is captured.

3. **Optimized Tokenization**:
    - *Challenge*: Direct string chunking may not consider word or sentence boundaries.
    - *Solution*: By tokenizing the text first and then creating chunks based on token count, the implementation ensures that words or sentences aren't arbitrarily split, preserving their meaning.

4. **Weighted Job Requirements**:
    - *Challenge*: Some job requirements might be more important than others.
    - *Solution*: The code multiplies the "must_have" and "should_have" requirements by 2, effectively giving them more weight in the embedding process. This highlights the importance of these requirements in the similarity calculations.

5. **Efficient Similarity Calculation**:
    - *Challenge*: Computing cosine similarity in a naive way can be computationally expensive.
    - *Solution*: The use of `util.pytorch_cos_sim` provides an efficient way to compute cosine similarities using PyTorch, thereby speeding up the overall process.

6. **Scalable Ranking System**:
    - *Challenge*: With many resumes, establishing a ranking system can become challenging.
    - *Solution*: The pandas `.rank()` function is employed to efficiently rank resumes based on their similarity scores, ensuring the solution remains scalable as the dataset size grows.
