import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import { GoogleGenAI } from "@google/genai";

// --- Shared types ---

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ModelOption {
  id: string;
  name: string;
  provider: string;
}

export interface ProviderConfig {
  provider: string;
  models: ModelOption[];
  chat(model: string, messages: ChatMessage[]): Promise<string>;
}

// --- Azure Anthropic (via AI Foundry) ---

function createAzureAnthropicProvider(): ProviderConfig | null {
  const apiKey = process.env.AZURE_ANTHROPIC_API_KEY;
  const baseURL = process.env.AZURE_ANTHROPIC_BASE_URL; // e.g. https://ShaheerTestingProject.openai.azure.com/anthropic
  const deploymentName = process.env.AZURE_ANTHROPIC_DEPLOYMENT || "claude-opus-4-6";

  if (!apiKey || !baseURL) return null;

  return {
    provider: "azure-anthropic",
    models: [
      { id: deploymentName, name: `Claude Opus 4.6 (Azure)`, provider: "azure-anthropic" },
    ],
    async chat(model: string, messages: ChatMessage[]): Promise<string> {
      // Azure Foundry endpoint format:
      // {baseURL}/deployments/{deployment}/messages?api-version=2023-06-01-preview
      const url = `${baseURL.replace(/\/+$/, "")}/messages`;
		console.log("Azure URL:", url);

      const response = await fetch(url, {
        method: "POST",
        headers: {
					"Content-Type": "application/json",
					"Authorization": `Bearer ${apiKey}`,
					"anthropic-version": "2023-06-01",
		},
        body: JSON.stringify({
			model: deploymentName,
          max_tokens: 4096,
          system: "You are a helpful Ponsse field assistant. Answer clearly and concisely.",
          messages: messages.map((m) => ({
            role: m.role,
            content: m.content,
          })),
        }),
      });

      if (!response.ok) {
        const err = await response.text();
        throw new Error(`Azure Anthropic error ${response.status}: ${err}`);
      }

      const data = await response.json();
      const block = data?.content?.[0];
      if (!block || block.type !== "text") {
        throw new Error("No response from Azure Anthropic");
      }
      return block.text;
    },
  };
}

// --- OpenAI / LM Studio ---

function createOpenAIProvider(): ProviderConfig | null {
  const apiKey = process.env.OPENAI_API_KEY;
  const baseURL = process.env.OPENAI_BASE_URL;

  // Only register if explicitly configured
  if (!apiKey && !baseURL) return null;

  const client = new OpenAI({
    apiKey: apiKey || "lm-studio",
    baseURL: baseURL || "http://127.0.0.1:1234/v1",
  });

  return {
    provider: "openai",
    models: [
      {
        id: "mistral-7b-instruct",
        name: "Mistral 7B (LM Studio)",
        provider: "openai",
      },
    ],
    async chat(model: string, messages: ChatMessage[]): Promise<string> {
      const completion = await client.chat.completions.create({
        model: "mistral-7b-instruct",
        messages,
        temperature: 0.7,
      });

      const content = completion.choices[0]?.message?.content;
      if (!content) throw new Error("No response from LM Studio");
      return content;
    },
  };
}

// --- Anthropic direct (non-Azure) ---

function createAnthropicProvider(): ProviderConfig | null {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) return null;

  const client = new Anthropic({ apiKey });

  return {
    provider: "anthropic",
    models: [
      { id: "claude-opus-4-6", name: "Claude Opus 4.6", provider: "anthropic" },
      { id: "claude-sonnet-4-5-20250929", name: "Claude Sonnet 4.5", provider: "anthropic" },
      { id: "claude-haiku-4-5-20251001", name: "Claude Haiku 4.5", provider: "anthropic" },
    ],
    async chat(model: string, messages: ChatMessage[]): Promise<string> {
      const response = await client.messages.create({
        model,
        max_tokens: 4096,
        system: "You are a helpful assistant.",
        messages: messages.map((m) => ({
          role: m.role,
          content: m.content,
        })),
      });

      const block = response.content[0];
      if (!block || block.type !== "text") {
        throw new Error("No response from Anthropic");
      }
      return block.text;
    },
  };
}

// --- Google Gemini ---

function createGeminiProvider(): ProviderConfig | null {
  const apiKey = process.env.GOOGLE_AI_API_KEY;
  if (!apiKey) return null;

  const client = new GoogleGenAI({ apiKey });

  return {
    provider: "gemini",
    models: [
      { id: "gemini-2.5-pro", name: "Gemini 2.5 Pro", provider: "gemini" },
      { id: "gemini-2.5-flash", name: "Gemini 2.5 Flash", provider: "gemini" },
      { id: "gemini-2.5-flash-lite", name: "Gemini 2.5 Flash-Lite", provider: "gemini" },
      { id: "gemini-2.0-flash", name: "Gemini 2.0 Flash", provider: "gemini" },
    ],
    async chat(model: string, messages: ChatMessage[]): Promise<string> {
      const contents = messages.map((m) => ({
        role: m.role === "assistant" ? ("model" as const) : ("user" as const),
        parts: [{ text: m.content }],
      }));

      const response = await client.models.generateContent({
        model,
        contents,
        config: { systemInstruction: "You are a helpful assistant." },
      });

      const text = response.text;
      if (!text) throw new Error("No response from Gemini");
      return text;
    },
  };
}

// --- AiWo RAG backend (Aalto) ---
// Routes chat through the Aalto-hosted visual RAG backend, which retrieves
// relevant manual pages and generates an answer with a VLM (Azure/Claude/local).

function createAiwoRagProvider(): ProviderConfig | null {
  const baseUrl = process.env.AIWO_RAG_BASE_URL;
  if (!baseUrl) return null;

  const modelId = process.env.AIWO_RAG_MODEL_ID || "aiwo-rag";
  const modelName = process.env.AIWO_RAG_MODEL_NAME || "AiWo RAG (Ponsse)";

  return {
    provider: "aiwo-rag",
    models: [{ id: modelId, name: modelName, provider: "aiwo-rag" }],
    async chat(model: string, messages: ChatMessage[]): Promise<string> {
      const url = `${baseUrl.replace(/\/+$/, "")}/chat`;

      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: messages.map((m) => ({ role: m.role, content: m.content })),
          provider: "aiwo-rag",
          model,
        }),
      });

      if (!response.ok) {
        const err = await response.text();
        throw new Error(`AiWo RAG error ${response.status}: ${err}`);
      }

      const data = await response.json();
      const content = data?.result?.content;
      if (typeof content !== "string" || content.length === 0) {
        throw new Error("No response from AiWo RAG backend");
      }
      return content;
    },
  };
}

// --- Ollama (local models) ---

function createOllamaProvider(): ProviderConfig | null {
  const baseUrl = process.env.OLLAMA_BASE_URL;
  if (!baseUrl) return null;

  const defaultModels = (process.env.OLLAMA_MODELS || "llama3.3,mistral,phi4")
    .split(",")
    .map((m) => m.trim())
    .filter(Boolean);

  return {
    provider: "ollama",
    models: defaultModels.map((m) => ({
      id: m,
      name: `${m} (local)`,
      provider: "ollama",
    })),
    async chat(model: string, messages: ChatMessage[]): Promise<string> {
      const url = `${baseUrl.replace(/\/+$/, "")}/api/chat`;

      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          messages: [
            { role: "system", content: "You are a helpful assistant." },
            ...messages,
          ],
          stream: false,
        }),
      });

      if (!response.ok) throw new Error(`Ollama returned ${response.status}`);

      const data = await response.json();
      const content = data?.message?.content;
      if (!content) throw new Error("No response from Ollama");
      return content;
    },
  };
}

// --- Provider registry ---
// Azure Anthropic is first — it takes priority if configured

const providerFactories = [
  createAzureAnthropicProvider,
  createAnthropicProvider,
  createAiwoRagProvider,
  createOpenAIProvider,
  createGeminiProvider,
  createOllamaProvider,
];

let cachedProviders: ProviderConfig[] | null = null;

export function getProviders(): ProviderConfig[] {
  if (!cachedProviders) {
    cachedProviders = providerFactories
      .map((factory) => factory())
      .filter((p): p is ProviderConfig => p !== null);
  }
  return cachedProviders;
}

export function getAvailableModels(): ModelOption[] {
  return getProviders().flatMap((p) => p.models);
}

export function findProvider(
  providerName: string,
): ProviderConfig | undefined {
  return getProviders().find((p) => p.provider === providerName);
}
