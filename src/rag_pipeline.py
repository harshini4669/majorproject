from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from typing import List
from src.retriever import FlightRetriever


class ADSBRAGPipeline:
    """Core RAG pipeline for ADS-B flight data queries"""
    
    def __init__(self, groq_api_key: str, retriever: FlightRetriever, model: str = "openai/gpt-oss-20b"):
        """
        Initialize RAG pipeline with Groq LLM
        
        Args:
            groq_api_key: API key for Groq
            retriever: FlightRetriever instance
            model: Groq model to use
        """
        self.retriever = retriever
        self.model = model
        
        # Initialize Groq LLM
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model=model,
            temperature=0.3,
            max_tokens=1024
        )
        
        # Define system prompt for ADS-B context
        self.system_prompt = """You are an expert aviation assistant specializing in ADS-B (Automatic Dependent Surveillance-Broadcast) flight data analysis. 

Your role is to:
1. Answer questions about flight positions, altitudes, speeds, and routes
2. Provide insights based on real-time ADS-B data
3. Identify patterns and anomalies in flight data
4. Explain aviation terminology clearly
5. Always cite the specific flight data when answering queries

When answering:
- Be precise with numbers (altitudes, speeds, coordinates)
- Use aviation terminology correctly
- Flag any unusual flight patterns
- Provide context for your answers based on the retrieved data

If the retrieved data doesn't contain relevant information, say so clearly."""
        
        # Define prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Based on the following flight data context, answer the user's question:

Context:
{context}

Question: {question}""")
        ])
        
        # Build RAG chain
        self.chain = (
            {
                "context": self._get_context,
                "question": RunnablePassthrough()
            }
            | self.prompt_template
            | self.llm
        )
    
    def _get_context(self, query: str) -> str:
        """Retrieve context for a given query"""
        return self.retriever.get_augmented_context(query, k=5)
    
    def query(self, question: str) -> dict:
        """
        Process a query through the RAG pipeline
        
        Args:
            question: User question about flights
            
        Returns:
            Dictionary with question, context, and answer
        """
        # Retrieve context
        context = self.retriever.get_augmented_context(question, k=5)
        
        # Generate response using LLM
        try:
            # Invoke the chain
            response = self.llm.invoke(
                self.prompt_template.format_messages(
                    context=context,
                    question=question
                )
            )
            
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "question": question,
                "context": context,
                "answer": answer,
                "success": True
            }
        except Exception as e:
            return {
                "question": question,
                "context": context,
                "answer": f"Error generating response: {str(e)}",
                "success": False
            }

    def run_query(self, user_query: str) -> str:
        """Return just the answer text for one question (used by the /ask API)."""
        result = self.query(user_query)
        return result.get("answer", "")

    def chat(self, questions: List[str]) -> List[dict]:
        """
        Process multiple questions in sequence
        
        Args:
            questions: List of questions
            
        Returns:
            List of response dictionaries
        """
        responses = []
        for question in questions:
            response = self.query(question)
            responses.append(response)
        
        return responses
    
    def interactive_chat(self):
        """Run interactive chatbot mode"""
        print("\n" + "=" * 70)
        print("ADS-B Flight Data RAG Chatbot")
        print("=" * 70)
        print("Ask me anything about the flight data!")
        print("Type 'exit' or 'quit' to end the conversation.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit']:
                    print("\nThank you for using the ADS-B Chatbot. Goodbye!")
                    break
                
                print("\nProcessing your query...\n")
                response = self.query(user_input)
                
                print(f"Assistant: {response['answer']}")
                print("\n" + "-" * 70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nChatbot interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")
