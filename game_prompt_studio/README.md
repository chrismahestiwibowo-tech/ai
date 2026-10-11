# Game Prompt Studio

A bilingual (English / Bahasa Indonesia) local web app for writing prompts for **Gemini Canvas**. Use the IDN / ENG switch at the top right to change the interface language, and the "Prompt language" option to write the prompt in either language. You pick the game type, theme, players, features and look, and the app writes a **STAR** prompt (Situation, Task, Action, Result). Gemini Canvas uses that prompt to build a complete HTML/CSS/JS game.

## Run on localhost

```sh
cd "prompt generator"
python3 -m http.server 8000
```

Then open http://localhost:8000 in your browser. You can also double-click `index.html`. It needs no install and no dependencies.

## Use

1. Fill in the form, or click one of the "Start from" presets.
2. The prompt on the right updates as you type. Click **Copy prompt**.
3. Click **Open Gemini**, turn on **Canvas**, paste the prompt and send it.
4. Play the game in the Canvas preview. Use the follow-up prompts in the app to keep improving it.
5. To run the game on localhost, save the code Gemini gives you as `index.html` in its own folder and serve it with the same `python3 -m http.server` command.

## Files

- `index.html`: page layout and form
- `style.css`: styles (light and dark mode)
- `app.js`: game types, features, presets and the STAR prompt builder (`buildPrompt`)

To add a new game type, add an entry to `GENRES` in `app.js`. To add a feature checkbox, add an entry to `FEATURES`.
