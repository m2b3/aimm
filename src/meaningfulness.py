import asyncio
import base64
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional, Protocol, Tuple

import pandas as pd
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic import BaseModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)

# Default prompt ID constant
DEFAULT_PROMPT_ID = 'p00000'


def _get_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == '.png':
        return 'image/png'
    elif suffix in ['.jpg', '.jpeg']:
        return 'image/jpeg'
    elif suffix == '.webp':
        return 'image/webp'
    else:
        return 'image/png'



class ImageMeaningfulness(BaseModel):
    reasoning: Optional[str] = None
    score: float
    confidence: Optional[float] = None
    model: str  # Track which model generated this
    prompt_content: str  # Full text of the prompt used
    prompt_id: str  # ID of the prompt used

    def model_dump(self, *args, **kwargs):
        d = super().model_dump(*args, **kwargs)
        return {k: v for k, v in d.items() if v is not None}

class Provider(Protocol):
    async def analyze_image(
        self,
        image_path: Path,
        prompt: str,
        prompt_id: str,
        include_reasoning: bool = True,
        temperature: Optional[float] = None,
        reference_images: Optional[list[Path]] = None,
    ) -> Optional[ImageMeaningfulness]: ...

class OpenRouterGeminiProvider:
    """Gemini provider using OpenRouter API with OpenAI-compatible interface."""
    
    def __init__(self, api_key: str, model: str = "google/gemini-2.5-flash", temperature: float = 0.5):
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model
        self.temperature = temperature
        self._cached_reference_images = None

    def _load_and_cache_reference_images(self, reference_images: list[Path]) -> list[dict]:
        """Load reference images once and cache them for reuse."""
        if self._cached_reference_images is None:
            cached_images = []
            for ref_img in reference_images:
                ref_content = ref_img.read_bytes()
                ref_b64 = base64.b64encode(ref_content).decode('utf-8')
                ref_media_type = _get_media_type(ref_img)
                ref_image_url = f"data:{ref_media_type};base64,{ref_b64}"
                cached_images.append({
                    "type": "image_url",
                    "image_url": {
                        "url": ref_image_url
                    },
                })
            
            # Add cache control to last reference image
            if cached_images:
                cached_images[-1]["cache_control"] = {
                    "type": "ephemeral"
                }
            
            self._cached_reference_images = cached_images
        
        return self._cached_reference_images
    
    def _parse_prompt_with_images(self, prompt: str, reference_images: Optional[list[Path]] = None) -> list[dict]:
        """Parse prompt and interleave text with reference images at appropriate positions."""
        content = []
        
        if reference_images:
            cached_ref_images = self._load_and_cache_reference_images(reference_images)
            
            # Split prompt by image placeholders and interleave with actual images
            parts = prompt.split('\n')
            ref_image_idx = 0
            
            current_text = ""
            for part in parts:
                # Check if this line is an image placeholder (format: "filename.ext" - description)
                if '" - ' in part and part.strip().startswith('"') and any(ext in part for ext in ['.png', '.jpg', '.jpeg']):
                    # Add accumulated text before the image
                    if current_text.strip():
                        content.append({
                            "type": "text",
                            "text": current_text.strip()
                        })
                        current_text = ""
                    
                    # Add the reference image if available
                    if ref_image_idx < len(cached_ref_images):
                        content.append(cached_ref_images[ref_image_idx])
                        ref_image_idx += 1
                    
                    # Extract description part after " - "
                    desc_part = part.split('" - ', 1)
                    if len(desc_part) > 1:
                        current_text += desc_part[1] + "\n"
                else:
                    current_text += part + "\n"
            
            # Add any remaining text
            if current_text.strip():
                content.append({
                    "type": "text",
                    "text": current_text.strip()
                })
        else:
            # No reference images, just add the prompt as text
            content.append({
                "type": "text",
                "text": prompt
            })
        
        return content

    async def analyze_image(
        self,
        image_path: Path,
        prompt: str,
        prompt_id: str,
        include_reasoning: bool = True,
        temperature: Optional[float] = None,
        reference_images: Optional[list[Path]] = None,
    ) -> Optional[ImageMeaningfulness]:
        try:
            # Use per-call temperature or fall back to default
            actual_temperature = temperature if temperature is not None else self.temperature

            # Parse prompt and interleave with reference images
            content = self._parse_prompt_with_images(prompt, reference_images)

            # Add main image at the end
            main_content = image_path.read_bytes()
            main_b64 = base64.b64encode(main_content).decode('utf-8')
            main_media_type = _get_media_type(image_path)
            main_image_url = f"data:{main_media_type};base64,{main_b64}"
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": main_image_url
                }
            })

            messages = [{
                "role": "user",
                "content": content
            }]
            # log.info("Messages:", messages)
            # raise RuntimeError("Debug stop")

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body = {
                    "usage": {"include": True},
                    "reasoning": {
                        # "enabled": False,
                        "max_tokens": 128,
                    }
                },
                # usage={
                # "include": True
                # },
                temperature=actual_temperature,
                max_tokens=1024,
                extra_headers={}
            )

            log.info(f"OpenRouter response: {response.usage}")

            # Parse JSON response
            response_text = response.choices[0].message.content
            if response_text is None:
                raise ValueError(f"Model returned null content (possible safety filter or empty response) for {image_path.name}")

            # Try to extract JSON from response (in case model wraps it in markdown or other text)
            try:
                # First try direct JSON parsing
                data = json.loads(response_text)
            except json.JSONDecodeError:
                # If that fails, try to find JSON within the response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError(f"Could not parse JSON from response: {response_text}")

            # Validate required fields
            if 'score' not in data:
                raise ValueError(f"Missing required 'score' field in response: {data}")
            
            if include_reasoning and 'reasoning' not in data:
                raise ValueError(f"Missing reasoning field in response: {data}")

            # Add metadata
            data['model'] = self.model
            data['prompt_content'] = prompt
            data['prompt_id'] = prompt_id
            
            return ImageMeaningfulness.model_validate(data)

        except Exception as e:
            log.error(f"OpenRouter Gemini API error for {image_path}: {e}")
            return None

class LocalVLLMProvider:
    """Local vLLM provider using the OpenAI-compatible chat completions API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "qwen3.5-27b",
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 300.0,
    ):
        self.client = AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            timeout=timeout,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._cached_reference_images = None

    def _load_and_cache_reference_images(self, reference_images: list[Path]) -> list[dict]:
        """Load reference images once and cache OpenAI-compatible image parts."""
        if self._cached_reference_images is None:
            cached_images = []
            for ref_img in reference_images:
                ref_b64 = base64.b64encode(ref_img.read_bytes()).decode('utf-8')
                ref_media_type = _get_media_type(ref_img)
                cached_images.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{ref_media_type};base64,{ref_b64}"
                    },
                })
            self._cached_reference_images = cached_images

        return self._cached_reference_images

    def _parse_prompt_with_images(self, prompt: str, reference_images: Optional[list[Path]] = None) -> list[dict]:
        """Parse prompt placeholders and interleave reference images."""
        content = []

        if reference_images:
            cached_ref_images = self._load_and_cache_reference_images(reference_images)
            parts = prompt.split('\n')
            ref_image_idx = 0
            current_text = ""

            for part in parts:
                if '" - ' in part and part.strip().startswith('"') and any(ext in part for ext in ['.png', '.jpg', '.jpeg']):
                    if current_text.strip():
                        content.append({
                            "type": "text",
                            "text": current_text.strip()
                        })
                        current_text = ""

                    if ref_image_idx < len(cached_ref_images):
                        content.append(cached_ref_images[ref_image_idx])
                        ref_image_idx += 1

                    desc_part = part.split('" - ', 1)
                    if len(desc_part) > 1:
                        current_text += desc_part[1] + "\n"
                else:
                    current_text += part + "\n"

            if current_text.strip():
                content.append({
                    "type": "text",
                    "text": current_text.strip()
                })
        else:
            content.append({
                "type": "text",
                "text": prompt
            })

        return content

    def _parse_score_response(self, response_text: str) -> dict:
        """Strictly parse local model output as {"score": <number 1..6>}."""
        import re

        text = response_text.strip()
        # Remove explicit reasoning blocks some models emit before the final answer.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip() or text

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            # Some models wrap JSON in markdown fences or extra text.
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError as inner_exc:
                    raise ValueError(f"Local vLLM returned unparsable JSON response: {response_text}") from inner_exc
            else:
                # Last-resort fallback: parse a plain-text score mention.
                score_match = re.search(r'\bscore\b\s*(?:is|=|:)?\s*([1-6](?:\.\d+)?)\b', text, re.IGNORECASE)
                if not score_match:
                    score_match = re.search(r'\b([1-6](?:\.\d+)?)\b', text)
                if not score_match:
                    raise ValueError(f"Local vLLM returned non-JSON response: {response_text}") from exc
                data = {"score": float(score_match.group(1))}

        if not isinstance(data, dict):
            raise ValueError(f"Local vLLM response must be a JSON object: {data}")
        if "score" not in data:
            raise ValueError(f"Local vLLM response must contain 'score': {data}")

        score = data["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"Local vLLM score must be numeric: {data}")

        score = float(score)
        if not 1.0 <= score <= 6.0:
            raise ValueError(f"Local vLLM score must be in [1, 6]: {data}")

        return {"score": score}

    async def analyze_image(
        self,
        image_path: Path,
        prompt: str,
        prompt_id: str,
        include_reasoning: bool = True,
        temperature: Optional[float] = None,
        reference_images: Optional[list[Path]] = None,
    ) -> Optional[ImageMeaningfulness]:
        try:
            actual_temperature = temperature if temperature is not None else self.temperature

            content = self._parse_prompt_with_images(prompt, reference_images)
            main_b64 = base64.b64encode(image_path.read_bytes()).decode('utf-8')
            main_media_type = _get_media_type(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{main_media_type};base64,{main_b64}"
                }
            })

            request_kwargs = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": content
                }],
                "temperature": actual_temperature,
                "max_tokens": self.max_tokens,
            }

            # qwen3 on vLLM may spend the whole budget on reasoning and return
            # null content unless thinking is disabled.
            extra_body = {}
            if not include_reasoning:
                extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                # Ask vLLM guided decoding to constrain model output to score JSON.
                extra_body["guided_json"] = {
                    "type": "object",
                    "properties": {
                        "score": {
                            "type": "number",
                            "minimum": 1,
                            "maximum": 6,
                        }
                    },
                    "required": ["score"],
                    "additionalProperties": False,
                }
                request_kwargs["response_format"] = {"type": "json_object"}

            if extra_body:
                request_kwargs["extra_body"] = extra_body

            response = await self.client.chat.completions.create(**request_kwargs)

            message = response.choices[0].message
            response_text = message.content
            if response_text is None:
                # Fallback for providers exposing reasoning text in a separate field.
                response_text = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
            if isinstance(response_text, list):
                response_text = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in response_text
                )
            if response_text is None:
                raise ValueError(f"Local vLLM returned null content for {image_path.name}")

            data = self._parse_score_response(response_text)
            data['model'] = self.model
            data['prompt_content'] = prompt
            data['prompt_id'] = prompt_id

            return ImageMeaningfulness.model_validate(data)

        except Exception as e:
            log.error(f"Local vLLM API error for {image_path}: {e}")
            return None

class GeminiProvider:
    """Legacy Gemini provider using direct Google API access."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-preview-04-17", temperature: float = 0.5):
        try:
            from google import genai
            self.client = genai.Client(api_key=api_key)
            self.model = model
            self.temperature = temperature
        except ImportError:
            raise ImportError("google-genai package is required for GeminiProvider. Install with: pip install google-genai")

    async def analyze_image(
        self,
        image_path: Path,
        prompt: str,
        prompt_id: str,
        include_reasoning: bool = True,
        temperature: Optional[float] = None,
        reference_images: Optional[list[Path]] = None,
    ) -> Optional[ImageMeaningfulness]:
        try:
            from google import genai

            # Use per-call temperature or fall back to default
            actual_temperature = temperature if temperature is not None else self.temperature

            schema = {
                'type': 'OBJECT',
                'properties': {
                    'score': {'type': 'NUMBER'},
                    'confidence': {'type': 'NUMBER'},
                    **(
                        {'reasoning': {'type': 'STRING'}}
                        if include_reasoning else {}
                    ),
                },
                'required': ['score'] + (
                    ['reasoning'] if include_reasoning else []
                )
            }

            # Build contents array starting with prompt
            contents = [prompt]
            
            # Add reference images first if provided
            if reference_images:
                for ref_img in reference_images:
                    ref_content = ref_img.read_bytes()
                    contents.append(genai.types.Part.from_bytes(ref_content, 'image/png'))
            
            # Add main image last
            main_content = image_path.read_bytes()
            contents.append(genai.types.Part.from_bytes(main_content, 'image/png'))

            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=schema,
                    temperature=actual_temperature
                )
            )

            data = json.loads(response.text)
            data['model'] = self.model
            data['prompt_content'] = prompt
            data['prompt_id'] = prompt_id
            return ImageMeaningfulness.model_validate(data)

        except Exception as e:
            log.error(f"Gemini API error for {image_path}: {e}")
            return None

class AnthropicProvider:
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307", temperature: float = 0.5):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature

    async def analyze_image(
        self,
        image_path: Path,
        prompt: str,
        prompt_id: str,
        include_reasoning: bool = True,
        temperature: Optional[float] = None,
        reference_images: Optional[list[Path]] = None,
    ) -> Optional[ImageMeaningfulness]:
        try:
            # Use per-call temperature or fall back to default
            actual_temperature = temperature if temperature is not None else self.temperature

            # Build content array starting with text prompt
            content = [{"type": "text", "text": prompt}]
            
            # Add reference images first if provided
            if reference_images:
                for ref_img in reference_images:
                    ref_content = ref_img.read_bytes()
                    ref_b64 = base64.b64encode(ref_content).decode('utf-8')
                    ref_media_type = f"image/{ref_img.suffix.lower()[1:]}"
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": ref_media_type,
                            "data": ref_b64
                        }
                    })
            
            # Add main image last
            main_content = image_path.read_bytes()
            main_b64 = base64.b64encode(main_content).decode('utf-8')
            main_media_type = f"image/{image_path.suffix.lower()[1:]}"
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": main_media_type,
                    "data": main_b64
                }
            })

            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                temperature=actual_temperature,
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )

            data = json.loads(response.content[0].text)
            data['model'] = self.model
            data['prompt_content'] = prompt
            data['prompt_id'] = prompt_id
            return ImageMeaningfulness.model_validate(data)

        except Exception as e:
            log.error(f"Anthropic API error for {image_path}: {e}")
            return None

def _ensure_dataframe_compatibility(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has all required columns for backward compatibility."""
    required_columns = {
        'image_path': str,
        'image_hash': str, 
        'timestamp': 'datetime64[ns]',
        'reasoning': str,
        'score': float,
        'confidence': float,
        'model': str,
        'prompt_content': str,
        'prompt_id': str
    }
    
    for col, dtype in required_columns.items():
        if col not in df.columns:
            if col == 'prompt_content':
                # For backward compatibility, fill with empty string
                df[col] = ""
            elif col == 'prompt_id':
                df[col] = DEFAULT_PROMPT_ID
            else:
                df[col] = None
    
    return df

def _compute_prompt_hash(prompt: str) -> str:
    """Compute SHA-256 hash of prompt content for cache validation."""
    return hashlib.sha256(prompt.encode('utf-8')).hexdigest()

def _validate_cache_consistency(df: pd.DataFrame, prompt: str, prompt_id: str) -> None:
    """Validate that cached entries with same prompt_id have consistent prompt content.
    
    Args:
        df: DataFrame with cached results
        prompt: Current prompt content
        prompt_id: Current prompt ID
        
    Raises:
        ValueError: If inconsistent prompt content found for the same prompt_id
    """
    if df.empty:
        return
    
    # Get current prompt hash
    current_hash = _compute_prompt_hash(prompt)
    
    # Check entries with same prompt_id
    matching_entries = df[df['prompt_id'] == prompt_id]
    if matching_entries.empty:
        return
    
    # Validate consistency for entries with non-empty prompt_content
    non_empty_content = matching_entries[matching_entries['prompt_content'].str.len() > 0]
    if non_empty_content.empty:
        return
    
    # Check all entries have same prompt hash
    cached_hashes = set(non_empty_content['prompt_content'].apply(_compute_prompt_hash))
    if len(cached_hashes) > 1:
        raise ValueError(
            f"Inconsistent prompt content found for prompt_id '{prompt_id}'. "
            f"Found {len(cached_hashes)} different prompt hashes in cache. "
            f"This indicates the same prompt_id was used with different prompts."
        )
    
    # Check current prompt matches cached prompt
    cached_hash = next(iter(cached_hashes))
    if current_hash != cached_hash:
        raise ValueError(
            f"Current prompt content doesn't match cached prompt for prompt_id '{prompt_id}'. "
            f"Current hash: {current_hash[:8]}..., Cached hash: {cached_hash[:8]}... "
            f"Either use a different prompt_id or ensure prompt content is identical."
        )

async def process_images(
    image_dir: Path,
    provider: Provider,
    output_path: Path,
    prompt: str,
    prompt_id: str,
    *,
    include_reasoning: bool = True,
    max_concurrent: int = 3,
    rate_limit: float = 6.0,
    limit: Optional[int] = None,
    temperature: Optional[float] = None,
    reference_images: Optional[list[Path]] = None,
) -> AsyncIterator[dict]:
    """Process images sequentially, with caching and rate limiting.

    WARNING: rate_limit is interpreted as SLEEP SECONDS between calls here,
    but as CALLS PER SECOND in process_images_parallel. The parallel path
    is what notebook 2 uses (use_parallel=True by default).
    """

    sem = asyncio.Semaphore(max_concurrent)

    # Track processed images by filename so identical patches from different scenes each get an entry
    processed_paths = set()
    hash_to_result = {}  # (hash, model, prompt_id) -> cached result dict for reuse
    if output_path.exists():
        df = pd.read_parquet(output_path)
        df = _ensure_dataframe_compatibility(df)

        # Validate cache consistency for prompt content
        _validate_cache_consistency(df, prompt, prompt_id)

        processed_paths = set(zip(df['image_path'].apply(lambda p: Path(p).name), df['model'], df['prompt_id']))
        log.info(f"Found {len(processed_paths)} previously processed images")

        # Build hash lookup to propagate scores to identical patches without re-calling the API
        for _, row in df.iterrows():
            key = (row['image_hash'], row['model'], row['prompt_id'])
            if key not in hash_to_result:
                hash_to_result[key] = row.to_dict()

    # Get all PNG paths and their hashes
    paths_and_hashes = [
        (p, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(image_dir.glob("**/*.png"))
    ]

    # Filter out already processed paths
    provider_model = provider.model
    paths_and_hashes = [
        (p, h) for p, h in paths_and_hashes
        if (p.name, provider_model, prompt_id) not in processed_paths
    ]

    # Apply limit after deduplication
    if limit is not None:
        paths_and_hashes = paths_and_hashes[:limit]

    total = len(paths_and_hashes)
    log.info(f"Processing {total} new images with prompt_id: {prompt_id}")

    for i, (path, img_hash) in enumerate(paths_and_hashes, 1):
        hash_key = (img_hash, provider_model, prompt_id)
        if hash_key in hash_to_result:
            # Propagate score to this path without calling the API
            cached = hash_to_result[hash_key]
            data = {**cached, 'image_path': str(path), 'timestamp': datetime.now()}
            log.info(f"Reusing cached result for identical patch {i}/{total}: {path}")
        else:
            async with sem:  # Rate limit
                await asyncio.sleep(rate_limit)

                log.info(f"Processing image {i}/{total}: {path}")
                result = await provider.analyze_image(
                    path,
                    include_reasoning=include_reasoning,
                    prompt=prompt,
                    prompt_id=prompt_id,
                    temperature=temperature,
                    reference_images=reference_images,
                )

                if not result:
                    log.warning(f"Failed to process {path}")
                    continue

                data = {
                    'image_path': str(path),
                    'image_hash': img_hash,
                    'timestamp': datetime.now(),
                    **result.model_dump()
                }
                hash_to_result[hash_key] = data  # Cache for reuse within this run

        # Save result
        existing_df = pd.DataFrame()
        if output_path.exists():
            existing_df = pd.read_parquet(output_path)
            existing_df = _ensure_dataframe_compatibility(existing_df)

        df = pd.concat([
            existing_df,
            pd.DataFrame([data])
        ], ignore_index=True)
        df.to_parquet(output_path, index=False)

        log.info(f"Saved result for {path}")
        yield data


async def process_images_parallel(
    image_dir: Path,
    provider: Provider,
    output_path: Path,
    prompt: str,
    prompt_id: str,
    *,
    include_reasoning: bool = True,
    max_concurrent: int = 3,
    rate_limit: float = 6.0,
    limit: Optional[int] = None,
    batch_size: int = 20,
    temperature: Optional[float] = None,
    reference_images: Optional[list[Path]] = None,
) -> AsyncIterator[dict]:
    # NOTE: rate_limit is calls-per-second here (inverse of process_images,
    # which treats it as sleep-seconds). This is the path used in production.
    class RateLimiter:
        def __init__(self, rate: float):
            self.rate = rate
            self.last_call = 0.0
        
        async def acquire(self):
            now = asyncio.get_event_loop().time()
            time_since_last = now - self.last_call
            if time_since_last < 1.0 / self.rate:
                await asyncio.sleep(1.0 / self.rate - time_since_last)
            self.last_call = asyncio.get_event_loop().time()
    
    rate_limiter = RateLimiter(rate_limit)
    sem = asyncio.Semaphore(max_concurrent)
    
    
    async def process_single_image(path: Path, img_hash: str, max_retries: int = 5) -> Optional[dict]:
        for attempt in range(max_retries + 1):
            async with sem:
                await rate_limiter.acquire()
                result = await provider.analyze_image(
                    path,
                    include_reasoning=include_reasoning,
                    prompt=prompt,
                    prompt_id=prompt_id,
                    temperature=temperature,
                    reference_images=reference_images,
                )
            if result:
                return {
                    'image_path': str(path),
                    'image_hash': img_hash,
                    'timestamp': datetime.now(),
                    **result.model_dump()
                }
            if attempt < max_retries:
                wait = 2 ** attempt
                log.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed for {path.name}, retrying in {wait}s")
                await asyncio.sleep(wait)
        log.error(f"All {max_retries + 1} attempts failed for {path.name} — patch will be missing from results")
        return None
    
    # Load existing data
    processed_paths = set()
    hash_to_result = {}  # (hash, model, prompt_id) -> cached result dict for reuse
    if output_path.exists():
        df = pd.read_parquet(output_path)
        df = _ensure_dataframe_compatibility(df)

        # Validate cache consistency for prompt content
        _validate_cache_consistency(df, prompt, prompt_id)

        processed_paths = set(zip(df['image_path'].apply(lambda p: Path(p).name), df['model'], df['prompt_id']))

        # Build hash lookup to propagate scores to identical patches without re-calling the API
        for _, row in df.iterrows():
            key = (row['image_hash'], row['model'], row['prompt_id'])
            if key not in hash_to_result:
                hash_to_result[key] = row.to_dict()

    # Get all PNG paths and their hashes
    paths_and_hashes = [
        (p, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(image_dir.glob("**/*.png"))
    ]

    # Filter out already processed paths
    provider_model = provider.model
    paths_and_hashes = [
        (p, h) for p, h in paths_and_hashes
        if (p.name, provider_model, prompt_id) not in processed_paths
    ]

    if limit:
        paths_and_hashes = paths_and_hashes[:limit]

    # Separate into: reuse from hash vs. need API call
    to_copy = [(p, h) for p, h in paths_and_hashes
               if (h, provider_model, prompt_id) in hash_to_result]
    to_process = [(p, h) for p, h in paths_and_hashes
                  if (h, provider_model, prompt_id) not in hash_to_result]

    log.info(f"Reusing results for {len(to_copy)} identical patches, processing {len(to_process)} new images with prompt_id: {prompt_id}")

    # Save copied results immediately
    if to_copy:
        copy_results = []
        for path, h in to_copy:
            cached = hash_to_result[(h, provider_model, prompt_id)]
            copy_results.append({**cached, 'image_path': str(path), 'timestamp': datetime.now()})

        existing_df = pd.DataFrame()
        if output_path.exists():
            existing_df = pd.read_parquet(output_path)
            existing_df = _ensure_dataframe_compatibility(existing_df)

        df = pd.concat([existing_df, pd.DataFrame(copy_results)], ignore_index=True)
        df.to_parquet(output_path, index=False)

        for data in copy_results:
            yield data

    if not to_process:
        return

    log.info(f"Processing {len(to_process)} images with prompt_id: {prompt_id}")

    # Process in batches
    for i in range(0, len(to_process), batch_size):
        batch = to_process[i:i + batch_size]

        if i == 0:
            # Process first image blocking to prime prompt cache, then
            # exclude it from the concurrent batch to avoid double-processing.
            first_path, first_hash = batch[0]
            log.info(f"Processing first image in a blocking fashion to ensure prompt caching and save money")
            first_result = await process_single_image(first_path, first_hash)
            if first_result is not None:
                # Save immediately so it doesn't get lost
                existing_df = pd.DataFrame()
                if output_path.exists():
                    existing_df = pd.read_parquet(output_path)
                    existing_df = _ensure_dataframe_compatibility(existing_df)
                df = pd.concat([existing_df, pd.DataFrame([first_result])], ignore_index=True)
                df.to_parquet(output_path, index=False)
                yield first_result
            batch = batch[1:]  # skip first image in concurrent batch

        # Create tasks for this batch
        tasks = [
            process_single_image(path, img_hash)
            for path, img_hash in batch
        ]

        # Execute batch concurrently
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter successful results
        successful_results = [
            r for r in batch_results
            if isinstance(r, dict) and r is not None
        ]

        # Save batch results
        if successful_results:
            existing_df = pd.DataFrame()
            if output_path.exists():
                existing_df = pd.read_parquet(output_path)
                existing_df = _ensure_dataframe_compatibility(existing_df)

            df = pd.concat([
                existing_df,
                pd.DataFrame(successful_results)
            ], ignore_index=True)
            df.to_parquet(output_path, index=False)

        # Yield results
        for result in successful_results:
            yield result


async def async_run_analysis(
    image_dir: Path,
    provider: Provider,
    output_path: Path,
    prompt: str,
    prompt_id: str,
    *,
    include_reasoning: bool = True,
    max_concurrent: int = 3,
    rate_limit: float = 6.0,
    limit: Optional[int] = None,
    batch_size: int = 20,
    use_parallel: bool = True,
    temperature: Optional[float] = None,
    reference_images: Optional[list[Path]] = None,
) -> list[dict]:
    """Convenience wrapper that collects all results.
    
    Args:
        image_dir: Directory containing images to process
        provider: AI provider instance (OpenRouterGeminiProvider, GeminiProvider, or AnthropicProvider)
        output_path: Path to save results parquet file
        include_reasoning: Whether to include reasoning in results
        max_concurrent: Maximum concurrent requests
        rate_limit: Rate limit for API calls
        limit: Maximum number of images to process
        prompt: Custom prompt text (optional)
        prompt_id: ID for the prompt (required if custom prompt provided)
        batch_size: Size of processing batches
        use_parallel: Whether to use parallel processing
        temperature: Temperature for model responses (optional, uses provider default if None)
        reference_images: List of reference images to include in prompt (optional)
        
    Returns:
        List of result dictionaries
    """
    if use_parallel:
        processor = process_images_parallel(
            image_dir, provider, output_path, prompt, prompt_id,
            include_reasoning=include_reasoning,
            max_concurrent=max_concurrent,
            rate_limit=rate_limit,
            limit=limit,
            batch_size=batch_size,
            temperature=temperature,
            reference_images=reference_images,
        )
    else:
        processor = process_images(
            image_dir, provider, output_path, prompt, prompt_id,
            include_reasoning=include_reasoning,
            max_concurrent=max_concurrent,
            rate_limit=rate_limit,
            limit=limit,
            temperature=temperature,
            reference_images=reference_images,
        )
    
    return [r async for r in processor]
