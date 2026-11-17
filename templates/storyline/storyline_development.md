# Video Storyline Development

Create a compelling demo video storyline for this website.

## Guidelines

- Target 3-5 minutes total duration
- Start with a strong first impression (homepage)
- Group related pages into logical chapters (2-5 chapters)
- Each chapter should tell a mini-story with a clear theme
- End with a call-to-action (visit the site, sign up, etc.)
- Use a natural narrative flow: introduce → explore → deep dive → conclude

## Storyline Structure

1. **Intro** (6-8 seconds): Title card with website name and tagline
2. **Chapters** (2-5): Each chapter covers a theme with 1-3 pages
3. **Outro** (5-6 seconds): Closing title card with CTA

## Output Format

Respond with a JSON object matching this schema:

```json
{
  "video_title": "Site Name — Tagline",
  "video_description": "A walkthrough of...",
  "target_duration_minutes": 4,
  "tone": "professional, friendly",
  "intro": {
    "title": "Welcome to Site Name",
    "subtitle": "Your solution for...",
    "duration_seconds": 8,
    "narrative_text": "Opening with..."
  },
  "chapters": [
    {
      "number": 1,
      "title": "Chapter Title",
      "subtitle": "Brief description",
      "pages": ["https://example.com/page1"],
      "narrative_arc": "What happens in this chapter",
      "key_moments": ["Moment 1", "Moment 2"],
      "estimated_duration_seconds": 60
    }
  ],
  "outro": {
    "title": "Get Started Today",
    "subtitle": "Visit example.com",
    "duration_seconds": 6
  }
}
```
