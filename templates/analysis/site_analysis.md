# Website Analysis Instructions

Analyze this website and produce a structured JSON response.

## Your Task

You are analyzing a website to prepare for creating a demo video. Examine the pages discovered below and produce a comprehensive understanding of:

1. **What is this website?** — Company/product/service description
2. **Who is the target audience?** — Personas, user types
3. **Key use cases** — 5-10 reasons someone visits this website
4. **Unique selling points** — What makes this stand out
5. **Narrative hooks** — Compelling angles for a demo video
6. **Tone and style** — Formal/casual, technical/consumer
7. **Visual identity** — Colors, brand elements detected

## Output Format

Respond with a JSON object:

```json
{
  "description": "One-paragraph description of the website",
  "target_audience": ["Persona 1", "Persona 2"],
  "use_cases": ["Use case 1", "Use case 2", ...],
  "selling_points": ["Point 1", "Point 2"],
  "narrative_hooks": ["Hook 1", "Hook 2"],
  "tone": "professional and friendly",
  "visual_identity": {
    "primary_color": "#hex",
    "style": "modern/minimal/corporate/playful"
  }
}
```
