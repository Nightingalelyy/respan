# respan-instrumentation-instructor

Respan instrumentation plugin for Instructor. This package instruments Instructor's native `patch()` and `Instructor.create()` paths directly and emits Respan-compatible chat spans without using `openinference-instrumentation-instructor`.

## Configuration

### 1. Install

```bash
pip install respan-instrumentation-instructor
```

### 2. Set Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `RESPAN_API_KEY` | Yes | Your Respan API key. Authenticates both proxy and tracing. |
| `RESPAN_BASE_URL` | No | Defaults to `https://api.respan.ai/api`. |

## Quickstart

### 3. Run Script

```python
import os

import instructor
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from respan import Respan
from respan_instrumentation_instructor import InstructorInstrumentor

load_dotenv()

respan_api_key = os.environ["RESPAN_API_KEY"]
respan_base_url = os.getenv("RESPAN_BASE_URL", "https://api.respan.ai/api")

# Route OpenAI traffic through the Respan gateway.
os.environ["OPENAI_API_KEY"] = respan_api_key
os.environ["OPENAI_BASE_URL"] = respan_base_url


class UserInfo(BaseModel):
    name: str
    age: int


respan = Respan(
    api_key=respan_api_key,
    base_url=respan_base_url,
    app_name="instructor-quickstart",
    instrumentations=[InstructorInstrumentor()],
)

client = instructor.from_openai(OpenAI())
user_info = client.create(
    response_model=UserInfo,
    messages=[{"role": "user", "content": "Ada Lovelace is 36 years old."}],
    model="gpt-4o-mini",
)

print(user_info.model_dump())
respan.flush()
```

### 4. View Dashboard

After running the script, traces appear on your [Respan dashboard](https://platform.respan.ai).

## Further Reading

Runnable examples with full setup instructions:

- **Python:** [python/tracing/instructor](https://github.com/respanai/respan-example-projects/tree/main/python/tracing/instructor)
