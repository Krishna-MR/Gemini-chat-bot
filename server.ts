import express from 'express';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const port = 3000;

app.use(express.json());

// API health / readiness status
app.get('/api/status', (req, res) => {
  res.json({
    status: 'ok',
    hasKey: Boolean(process.env.GEMINI_API_KEY),
    models: ['gemini-3.8-flash', 'gemini-4-argon', 'gemini-3.1-flash-lite'],
  });
});

// Streaming Chat endpoint using official @google/genai SDK
app.post('/api/chat', async (req, res) => {
  try {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return res.status(500).json({
        error: 'GEMINI_API_KEY environment variable is not set on the server.',
      });
    }

    const { contents, model = 'gemini-3.8-flash', systemInstruction } = req.body;

    if (!contents || !Array.isArray(contents)) {
      return res.status(400).json({
        error: 'Invalid contents format. Array of messages is required.',
      });
    }

    const ai = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const config: { systemInstruction?: string } = {};
    if (systemInstruction) {
      config.systemInstruction = systemInstruction;
    }

    // Helper to extract clean human-readable error from GenAI SDK
    function formatGenAIError(err: any): string {
      if (!err) return 'Unknown generation error';
      let raw = err.message || String(err);
      try {
        let parsed = typeof raw === 'string' && raw.trim().startsWith('{') ? JSON.parse(raw) : null;
        if (parsed?.error?.message) {
          try {
            let inner = JSON.parse(parsed.error.message);
            if (inner?.error?.message) return inner.error.message;
          } catch {}
          return parsed.error.message;
        }
      } catch {}
      return raw;
    }

    let responseStream;
    try {
      responseStream = await ai.models.generateContentStream({
        model,
        contents,
        config,
      });
    } catch (initialErr: any) {
      const errMsg = formatGenAIError(initialErr);
      return res.status(500).json({ error: errMsg });
    }

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    for await (const chunk of responseStream) {
      const text = chunk.text;
      if (text) {
        res.write(`data: ${JSON.stringify({ text })}\n\n`);
      }
    }

    res.write('data: [DONE]\n\n');
    res.end();
  } catch (error: any) {
    console.error('Gemini API stream error:', error);
    let message = error?.message || 'Failed to generate response';
    try {
      const parsed = JSON.parse(message);
      if (parsed?.error?.message) message = parsed.error.message;
    } catch {}

    if (res.headersSent) {
      res.write(`data: ${JSON.stringify({ error: message })}\n\n`);
      res.end();
    } else {
      res.status(500).json({ error: message });
    }
  }
});

// Mount Vite middleware for dev or serve static files in production
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.resolve(__dirname, 'dist')));
  app.get('*', (req, res) => {
    res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
  });
} else {
  const { createServer: createViteServer } = await import('vite');
  const vite = await createViteServer({
    server: { middlewareMode: true },
    appType: 'spa',
  });
  app.use(vite.middlewares);
}

app.listen(port, '0.0.0.0', () => {
  console.log(`Server running at http://0.0.0.0:${port}`);
});
