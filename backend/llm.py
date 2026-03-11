import os
from google import genai
from google.genai import types

class LLMInsightGenerator:
    def __init__(self, api_key: str = None):
        # Prefers environment variable GEMINI_API_KEY if not provided
        self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.model_id = "gemini-2.0-flash" # Use flash for speed and cost-efficiency

    def _build_system_instruction(self):
        return """You are an expert Senior Data Scientist and ML Architect. 
        Your task is to interpret technical EDA and ML diagnostic reports for a user.
        
        Guidelines:
        1. Summarize the 'Health' of the data first.
        2. Identify 'Showstoppers': If a model will fail (e.g., target leakage or near-zero variance), highlight it immediately.
        3. Explain technical metrics (like Silhouette scores or R2) in plain language.
        4. Provide clear 'Next Steps' (e.g., 'You should apply a log transform to feature X').
        5. Be concise but insightful. Avoid repeating every single number; focus on what the numbers MEAN."""

    async def generate_report_insights(self, diagnostic_results: dict):
        """
        Sends the JSON diagnostic report to Gemini and returns a structured markdown analysis.
        """
        prompt = f"""
        Please analyze the following ML Diagnostic Report and provide a human-readable executive summary.
        
        REPORT DATA:
        {diagnostic_results}
        
        STRUCTURE YOUR RESPONSE AS:
        ## Executive Summary
        ## Data Health & Quality
        ## Model Trainability & Risks
        ## Recommended Action Plan
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=self._build_system_instruction(),
                    temperature=0.2, # Lower temperature for more factual, stable reports
                )
            )
            return response.text
        except Exception as e:
            return f"Error generating insights: {str(e)}"