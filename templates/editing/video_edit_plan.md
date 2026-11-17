# Video Editing Plan

Analyze the recorded segments and suggest editing decisions.

For each recorded part, consider:

1. **Speed adjustment** — Should this part be sped up or slowed down?
   - Loading screens: speed up 2-3x
   - Form filling: keep at 1x
   - Scrolling: can speed up 1.5x
   - Important interactions: keep at 1x

2. **Trimming** — Should the start or end be trimmed?
   - Remove blank frames at start/end
   - Remove long loading waits
   - Keep transition moments

3. **Dead time** — Are there segments with no visual change?
   - Remove segments longer than 3s with no change
   - Replace with a brief fade or cut

Output a JSON array of edit decisions:
```json
[
  {
    "part_id": "ch01-part1",
    "speed_factor": 1.0,
    "trim_start_s": 0.5,
    "trim_end_s": 0.5,
    "notes": "Good pacing, just trim edges"
  }
]
```
