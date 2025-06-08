import os
import time
import asyncio
from collections import deque
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List

class GeminiSemanticFilter:
    """Gemini-based semantic filter for response text with rate-limited temporal queue."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini semantic filter."""
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            print("GEMINI_API_KEY not found - semantic filtering disabled")
            self.enabled = False
            return

        self.enabled = True
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemma-3n-e4b-it"  # Fast model for real-time
        self.previous_content = ""
        self.executor = ThreadPoolExecutor(max_workers=2)
        # Temporal queue to store (content, timestamp) tuples
        self.content_queue = deque()
        # Rate limiting: 30 calls per minute = 1 call every 2 seconds
        self.rate_limit_interval = 2.0  # seconds
        self.last_api_call = 0.0  # Timestamp of last API call

    def _generate_sync(self, prompt: str) -> str:
        """Synchronous generation for use with thread executor."""
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=100,  # Increased for summary + relevance
                    response_mime_type="text/plain",
                ),
            )
            
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API error: {e}")
            return ""

    async def filter_response(self, new_content: str) -> Tuple[bool, str]:
        """
        Compare new content with previous content for semantic differences, using a temporal queue
        to respect Gemini API rate limits (30 calls per minute). Returns new_content immediately if
        rate-limited, and Gemini-processed summary when API call is allowed.
        Returns (has_meaningful_change, filtered_response)
        """
        if not self.enabled:
            return True, new_content

        if not new_content or new_content.strip() == "":
            return False, ""

        # Add new content to queue with current timestamp
        current_time = time.time()
        self.content_queue.append((new_content, current_time))
        print("Time passed : ", current_time - self.last_api_call)
        # Check if we can make an API call (respecting 2-second interval)
        if current_time - self.last_api_call < self.rate_limit_interval:
            # Return new_content immediately while rate-limited
            return True, ""

        # Process the queue: summarize content and check relevance
        try:
            # Collect all queued content
            queued_contents: List[str] = [content for content, _ in self.content_queue]
            if not queued_contents:
                return True, new_content

            # Create a summary prompt
            summary_prompt = f"""Summarize the following sequence of texts into a concise description (1-10 words) of the meaningful changes or updates. The texts are outputs from deep learning-based scene descriptors/classifiers, describing a visual scene for a blind user. Focus on new objects, people, actions, environmental changes, or significant state updates critical for scene understanding. Ignore formatting, punctuation, or minor rephrasing.

Texts:
{chr(10).join(f"- {content}" for content in queued_contents)}

Summary:"""

            # Run summary generation in thread pool
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(
                self.executor,
                self._generate_sync,
                summary_prompt
            )

            if not summary or summary == "NO_CHANGE":
                self.content_queue.clear()
                self.last_api_call = current_time
                return True, new_content

            # Check if the summary is relevant to the latest content
            relevance_prompt = f"""You are assisting a blind user by filtering redundant scene information from deep learning-based scene descriptors/classifiers. These descriptors provide frequent updates (up to 30 FPS) about a visual scene. Your task is to determine if the summarized changes are relevant to the latest scene description to aid scene understanding.

Summary of changes: "{summary}"
Latest scene description: "{new_content}"

Rules:
1. If the summary contains meaningful information relevant to the latest scene description (e.g., new objects, people, actions, environmental changes, or significant state updates), respond with the summary.
2. If the summary describes outdated, redundant, or irrelevant changes compared to the latest scene description, respond with "DISCARD".
3. Prioritize information critical for a blind user, such as new obstacles, people, or significant scene changes, to ensure safe navigation and awareness.
4. Consider that the summary aggregates changes from multiple recent descriptions, but only return it if it adds value to understanding the latest scene.

Response:"""

            # Run relevance check in same API call window
            relevance_result = await loop.run_in_executor(
                self.executor,
                self._generate_sync,
                relevance_prompt
            )

            # Update last API call time
            self.last_api_call = current_time

            # Clear the queue and add the summary with the current timestamp
            self.content_queue.clear()
            if relevance_result != "DISCARD":
                self.content_queue.append((summary, current_time))
                self.previous_content = summary
                return True, summary
            else:
                self.previous_content = new_content
                return True, new_content

        except Exception as e:
            print(f"Error in semantic filtering: {e}")
            # On error, clear queue, update previous content, and pass through
            self.content_queue.clear()
            self.previous_content = new_content
            self.last_api_call = current_time
            return True, new_content

    def __del__(self):
        """Clean up thread pool executor."""
        self.executor.shutdown(wait=False)