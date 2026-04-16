import { useState, useRef, useEffect, useCallback, FormEvent, KeyboardEvent } from "react";
import Head from "next/head";
import styles from "../styles/Home.module.css";
import Image from "next/image";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import CircularProgress from "@mui/material/CircularProgress";
import Link from "next/link";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface ModelOption {
  id: string;
  name: string;
  provider: string;
}

type PTTState = "idle" | "listening" | "countdown" | "error";

export default function Home() {
  const [userInput, setUserInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Hi there! How can I help?" },
  ]);
  const [models, setModels] = useState<ModelOption[]>([]);
  const [selectedModel, setSelectedModel] = useState<ModelOption | null>(null);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [pttState, setPttState] = useState<PTTState>("idle");
  const [pttSupported, setPttSupported] = useState(false);
  const [micError, setMicError] = useState("");
  const [isSpeaking, setIsSpeaking] = useState(false);

  const recognitionRef = useRef<any>(null);
  const transcriptRef = useRef("");
  const countdownTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const selectedModelRef = useRef<ModelOption | null>(null);
  const messagesRef = useRef<Message[]>([]);
  const messageListRef = useRef<HTMLDivElement>(null);
  const textAreaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { selectedModelRef.current = selectedModel; }, [selectedModel]);
  useEffect(() => { messagesRef.current = messages; }, [messages]);

  useEffect(() => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    setPttSupported(!!SR);
  }, []);

  useEffect(() => {
    async function fetchModels() {
      try {
        const res = await fetch("/api/providers");
        if (res.ok) {
          const data = await res.json();
          setModels(data.models);
          if (data.models.length > 0) setSelectedModel(data.models[0]);
        }
      } catch {}
      finally { setModelsLoading(false); }
    }
    fetchModels();
  }, []);

  useEffect(() => {
    if (messageListRef.current)
      messageListRef.current.scrollTop = messageListRef.current.scrollHeight;
  }, [messages]);

  useEffect(() => {
    if (textAreaRef.current) textAreaRef.current.focus();
  }, []);

  // ── TTS: stop speaking immediately ──────────────────────────────────────────
  const stopSpeaking = useCallback(() => {
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
    }
  }, []);

  // ── TTS: speak assistant response ───────────────────────────────────────────
  const speak = useCallback((text: string) => {
    if (typeof window === "undefined" || !window.speechSynthesis) return;

    // Drop anything after the first horizontal-rule separator. The AiWo RAG
    // backend appends a "📄 Retrieved manual pages …" footer below a `---`
    // line — that's for the eyes, not the ears.
    const main = text.split(/\n\s*---\s*\n/)[0];

    // Strip markdown symbols so they aren't read aloud
    const clean = main
      .replace(/!\[.*?\]\(.+?\)/g, "")
      .replace(/#{1,6}\s/g, "")
      .replace(/\*\*(.+?)\*\*/g, "$1")
      .replace(/\*(.+?)\*/g, "$1")
      .replace(/`{1,3}[^`]*`{1,3}/g, "")
      .replace(/\[(.+?)\]\(.+?\)/g, "$1")
      .replace(/>\s.+/g, "")
      .replace(/[-*+]\s/g, "")
      .trim();

    window.speechSynthesis.cancel(); // cancel any previous speech
    const utterance = new SpeechSynthesisUtterance(clean);
    utterance.lang = "en-US";
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    window.speechSynthesis.speak(utterance);
  }, []);

  const handleError = (errorMessage?: string) => {
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: errorMessage || "Oops! There seems to be an error. Please try again." },
    ]);
    setLoading(false);
    setUserInput("");
  };

  const sendMessage = useCallback(async (text: string) => {
    const model = selectedModelRef.current;
    if (!text.trim() || !model) return;

    stopSpeaking(); // stop any ongoing TTS before sending
    setLoading(true);
    setUserInput("");
    setPttState("idle");
    setMicError("");

    const context: Message[] = [
      ...messagesRef.current,
      { role: "user", content: text },
    ];
    setMessages(context);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: context, provider: model.provider, model: model.id }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        handleError(
          response.status === 429
            ? "Too many requests. Please wait a moment and try again."
            : errorData?.error || "Something went wrong. Please try again."
        );
        return;
      }

      const data = await response.json();
      if (!data?.result?.content) { handleError(); return; }

      const assistantText = data.result.content;
      setMessages((prev) => [...prev, { role: "assistant", content: assistantText }]);

      // Auto-speak the assistant's response
      speak(assistantText);

    } catch {
      handleError("Network error. Please check your connection and try again.");
    } finally {
      setLoading(false);
    }
  }, [speak, stopSpeaking]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (userInput.trim() === "" || !selectedModel) return;
    await sendMessage(userInput);
  };

  const cancelCountdown = useCallback(() => {
    if (countdownTimerRef.current) {
      clearTimeout(countdownTimerRef.current);
      countdownTimerRef.current = null;
    }
    setPttState("idle");
    transcriptRef.current = "";
    setUserInput("");
  }, []);

  const stopListening = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.onend = null;
      recognitionRef.current.stop();
      recognitionRef.current = null;
    }

    const captured = transcriptRef.current.trim();
    if (!captured) {
      setPttState("idle");
      return;
    }

    setUserInput(captured);
    setPttState("countdown");

    countdownTimerRef.current = setTimeout(() => {
      countdownTimerRef.current = null;
      sendMessage(captured);
    }, 1500);
  }, [sendMessage]);

  const toggleMic = useCallback(() => {
    // Always stop TTS first — operator wants to speak, not listen to the answer
    stopSpeaking();

    if (pttState === "listening") {
      stopListening();
      return;
    }

    if (pttState === "countdown") {
      cancelCountdown();
      return;
    }

    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SR) return;

    transcriptRef.current = "";
    setMicError("");
    setUserInput("");

    const recognition = new SR();
    recognition.lang = "en-US";
    recognition.continuous = true;
    recognition.interimResults = true;

    recognition.onstart = () => {
      setPttState("listening");
    };

    recognition.onresult = (event: any) => {
      let interim = "";
      let final = "";
      for (let i = 0; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) {
          final += result[0].transcript;
        } else {
          interim += result[0].transcript;
        }
      }
      const display = (final + interim).trim();
      if (display) {
        transcriptRef.current = final || display;
        setUserInput(display);
      }
    };

    recognition.onerror = (event: any) => {
      recognitionRef.current = null;
      if (event.error === "no-speech") {
        setPttState("idle");
        setUserInput("");
      } else if (event.error === "not-allowed") {
        setPttState("error");
        setMicError("Mic blocked. Click the 🔒 icon in the address bar → allow microphone → refresh.");
        setTimeout(() => { setPttState("idle"); setMicError(""); }, 6000);
      } else {
        setPttState("error");
        setMicError(`Mic error: ${event.error}`);
        setTimeout(() => { setPttState("idle"); setMicError(""); }, 3000);
      }
    };

    recognition.onend = () => {
      if (recognitionRef.current) {
        recognitionRef.current = null;
        stopListening();
      }
    };

    recognitionRef.current = recognition;
    recognition.start();
  }, [pttState, stopListening, cancelCountdown, stopSpeaking]);

  const handleEnter = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Escape" && pttState === "countdown") { cancelCountdown(); return; }
    if (e.key === "Enter" && userInput) {
      if (!e.shiftKey) handleSubmit(e);
    } else if (e.key === "Enter") {
      e.preventDefault();
    }
  };

  const providerLabels: Record<string, string> = { ollama: "Ollama" };
  const groupedModels = models.reduce<Record<string, ModelOption[]>>((groups, model) => {
    const key = model.provider;
    if (!groups[key]) groups[key] = [];
    groups[key].push(model);
    return groups;
  }, {});

  const micLabel =
    pttState === "listening" ? "Tap to send" :
    pttState === "countdown" ? "Tap to cancel" :
    pttState === "error"     ? "Mic error" :
    isSpeaking               ? "Tap to interrupt" :
                               "Tap to speak";

  return (
    <>
      <Head>
        <title>Ponsse Assistant</title>
        <meta name="description" content="Multi-model AI chat interface" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <div className={styles.topnav}>
        <div className={styles.navlogo}>
          <Link href="/">Ponsse Assistant</Link>
        </div>
        <div className={styles.navlinks}>
          {modelsLoading ? (
            <span className={styles.modelloading}>Loading models...</span>
          ) : models.length === 0 ? (
            <span className={styles.modelloading}>No providers configured</span>
          ) : (
            <select
              className={styles.modelselect}
              value={selectedModel ? `${selectedModel.provider}:${selectedModel.id}` : ""}
              onChange={(e) => {
                const [provider, ...idParts] = e.target.value.split(":");
                const id = idParts.join(":");
                const model = models.find((m) => m.provider === provider && m.id === id);
                if (model) setSelectedModel(model);
              }}
              disabled={loading}
            >
              {Object.entries(groupedModels).map(([provider, providerModels]) => (
                <optgroup key={provider} label={providerLabels[provider] || provider}>
                  {providerModels.map((model) => (
                    <option key={`${model.provider}:${model.id}`} value={`${model.provider}:${model.id}`}>
                      {model.name}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          )}
        </div>
      </div>
      <main className={styles.main}>
        <div className={styles.cloud}>
          <div ref={messageListRef} className={styles.messagelist}>
            {messages.map((message, index) => (
              <div
                key={index}
                className={
                  message.role === "user" && loading && index === messages.length - 1
                    ? styles.usermessagewaiting
                    : message.role === "assistant"
                    ? styles.apimessage
                    : styles.usermessage
                }
              >
                {message.role === "assistant" ? (
                  <Image src="/openai.png" alt="AI" width="30" height="30" className={styles.boticon} priority />
                ) : (
                  <Image src="/usericon.png" alt="Me" width="30" height="30" className={styles.usericon} priority />
                )}
                <div className={styles.markdownanswer}>
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={{
                      a: ({ node, href, children, ...props }) => (
                        <a
                          href={href}
                          target="_blank"
                          rel="noopener noreferrer"
                          {...props}
                        >
                          {children}
                        </a>
                      ),
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className={styles.center}>
          <div className={styles.cloudform}>

            {/* Status banners */}
            {micError && (
              <div className={styles.micbanner}>{micError}</div>
            )}
            {pttState === "countdown" && !micError && (
              <div className={styles.micbanner} style={{ color: "#FFD200" }}>
                Sending in 1.5s — tap mic or press Esc to cancel
              </div>
            )}
            {isSpeaking && pttState === "idle" && (
              <div className={styles.micbanner} style={{ color: "#2ecc71" }}>
                🔊 Speaking — tap mic to interrupt
              </div>
            )}

            <form onSubmit={handleSubmit}>
              <div className={styles.inputrow}>
                <textarea
                  disabled={loading || models.length === 0}
                  onKeyDown={handleEnter}
                  ref={textAreaRef}
                  autoFocus={false}
                  rows={1}
                  maxLength={512}
                  id="userInput"
                  name="userInput"
                  placeholder={
                    models.length === 0
                      ? "No AI providers configured..."
                      : pttState === "listening"
                      ? "Listening… speak now"
                      : loading
                      ? "Waiting for response..."
                      : isSpeaking
                      ? "Tap mic to interrupt..."
                      : "Type or tap mic to speak..."
                  }
                  value={userInput}
                  onChange={(e) => {
                    if (pttState === "countdown") cancelCountdown();
                    setUserInput(e.target.value);
                  }}
                  className={styles.textarea}
                />

                {/* Mic toggle button */}
                {pttSupported && (
                  <button
                    type="button"
                    onClick={toggleMic}
                    className={[
                      styles.micbutton,
                      pttState === "listening"            ? styles.miclistening  : "",
                      pttState === "countdown"            ? styles.miccountdown  : "",
                      pttState === "error"                ? styles.micerror      : "",
                      isSpeaking && pttState === "idle"   ? styles.micspeaking   : "",
                    ].join(" ")}
                    aria-label={micLabel}
                    title={micLabel}
                    disabled={loading || models.length === 0}
                  >
                    {pttState === "listening" && (
                      <>
                        <span className={styles.micring} />
                        <span className={`${styles.micring} ${styles.micring2}`} />
                      </>
                    )}
                    {pttState === "countdown" && (
                      <span className={styles.miccountdownring} />
                    )}
                    {isSpeaking && pttState === "idle" && (
                      <>
                        <span className={styles.speakingring} />
                        <span className={`${styles.speakingring} ${styles.speakingring2}`} />
                      </>
                    )}
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      {pttState === "error" ? (
                        <>
                          <line x1="1" y1="1" x2="23" y2="23" />
                          <path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6" />
                          <path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23" />
                          <line x1="12" y1="19" x2="12" y2="23" />
                          <line x1="8" y1="23" x2="16" y2="23" />
                        </>
                      ) : (
                        <>
                          <rect x="9" y="1" width="6" height="11" rx="3" />
                          <path d="M19 10a7 7 0 0 1-14 0" />
                          <line x1="12" y1="19" x2="12" y2="23" />
                          <line x1="8" y1="23" x2="16" y2="23" />
                        </>
                      )}
                    </svg>
                  </button>
                )}

                {/* Send button */}
                <button
                  type="submit"
                  disabled={loading || models.length === 0}
                  className={styles.generatebutton}
                >
                  {loading ? (
                    <div className={styles.loadingwheel}>
                      <CircularProgress color="inherit" size={20} />
                    </div>
                  ) : (
                    <svg viewBox="0 0 20 20" className={styles.svgicon} xmlns="http://www.w3.org/2000/svg">
                      <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
                    </svg>
                  )}
                </button>
              </div>
            </form>
          </div>
          <div className={styles.footer}>
            <p>
              Powered by{" "}
              <a href="https://ollama.com/" target="_blank" rel="noopener noreferrer">Ollama</a>.
            </p>
          </div>
        </div>
      </main>
    </>
  );
}
