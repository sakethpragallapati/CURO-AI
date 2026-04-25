'use client';

import { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Brain, Send, X, RefreshCw, Activity, ArrowLeft, Sparkles, LogOut, Globe, Database } from 'lucide-react';
import { signOut } from 'firebase/auth';
import { auth } from '../lib/firebase';

export default function TriageAssistant() {
  const router = useRouter();
  const [triageMessages, setTriageMessages] = useState<{role: 'user' | 'assistant', content: string, options?: string[]}[]>([]);
  const [triageLoading, setTriageLoading] = useState(false);
  const [questionCount, setQuestionCount] = useState(0);
  const [isOtherSpecifying, setIsOtherSpecifying] = useState(false);
  const [triageInput, setTriageInput] = useState('');
  const [triageFinished, setTriageFinished] = useState(false);
  const [finalSummary, setFinalSummary] = useState('');
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [triageMessages, triageLoading]);

  // Start the triage on mount
  useEffect(() => {
    startTriage();
  }, []);

  const startTriage = async () => {
    setTriageLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/triage/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ history: [], question_count: 0 })
      });
      const data = await response.json();
      setTriageMessages([{
        role: 'assistant',
        content: data.question || "Hello! I'm Curo, your healthcare assistant. What brings you in today?",
        options: data.options || ["I have a specific symptom", "I feel unwell generally", "I need a health check", "Other (please specify)"]
      }]);
    } catch (err) {
      setTriageMessages([{
        role: 'assistant',
        content: "Hello! I'm Curo, your healthcare assistant. What brings you in today?",
        options: ["I have a specific symptom", "I feel unwell generally", "I need a health check", "Other (please specify)"]
      }]);
    } finally {
      setTriageLoading(false);
    }
  };

  const handleTriageSelect = async (optionLabel: string) => {
    const updatedMessages = [...triageMessages, { role: 'user' as const, content: optionLabel }];
    setTriageMessages(updatedMessages);

    if (optionLabel === "Other (please specify)") {
      setIsOtherSpecifying(true);
      setTriageMessages([...updatedMessages, { role: 'assistant', content: "Please tell me more about what you're experiencing:" }]);
      return;
    }

    setTriageLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/triage/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          history: updatedMessages.map(m => ({ role: m.role, content: m.content })),
          question_count: questionCount + 1
        })
      });
      const data = await response.json();

      if (data.finished) {
        const conclusion = data.summary || "Thank you for sharing. I have enough information to begin the analysis.";
        setTriageMessages([...updatedMessages, { role: 'assistant', content: conclusion }]);
        setTriageFinished(true);
        setFinalSummary(conclusion);
      } else {
        setTriageMessages([...updatedMessages, {
          role: 'assistant',
          content: data.question || "Could you tell me a little more about what you're feeling?",
          options: data.options || ["Other (please specify)"]
        }]);
        setQuestionCount(prev => prev + 1);
      }
    } catch (err) {
      console.error("Triage Agent Error:", err);
      setTriageMessages([...updatedMessages, {
        role: 'assistant',
        content: "I'm having trouble connecting. Could you try again?",
        options: ["Retry", "Other (please specify)"]
      }]);
    } finally {
      setTriageLoading(false);
    }
  };

  const handleOtherSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!triageInput.trim()) return;

    const userInput = triageInput.trim();
    const updatedMessages = [...triageMessages, { role: 'user' as const, content: userInput }];
    setTriageMessages(updatedMessages);
    setTriageInput('');
    setIsOtherSpecifying(false);
    setTriageLoading(true);

    try {
      const response = await fetch('http://localhost:8000/api/triage/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          history: updatedMessages.map(m => ({ role: m.role, content: m.content })),
          question_count: questionCount + 1
        })
      });
      const data = await response.json();

      if (data.finished) {
        const conclusion = data.summary || "Thank you. I have enough information to begin the analysis.";
        setTriageMessages([...updatedMessages, { role: 'assistant', content: conclusion }]);
        setTriageFinished(true);
        setFinalSummary(conclusion);
      } else {
        setTriageMessages([...updatedMessages, {
          role: 'assistant',
          content: data.question || "Could you tell me a little more about what you're feeling?",
          options: data.options || ["Other (please specify)"]
        }]);
        setQuestionCount(prev => prev + 1);
      }
    } catch (err) {
      console.error("Triage Agent Error:", err);
    } finally {
      setTriageLoading(false);
    }
  };

  const handleAnalyze = (selectedMode: 'generic' | 'deep-research') => {
    // Navigate to chat with the summary and selected mode
    const params = new URLSearchParams({ q: finalSummary, mode: selectedMode });
    router.push(`/chat?${params.toString()}`);
  };

  const handleRestart = () => {
    setTriageMessages([]);
    setQuestionCount(0);
    setTriageFinished(false);
    setFinalSummary('');
    setIsOtherSpecifying(false);
    startTriage();
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="sticky top-0 z-50 glass-card-strong border-t-0 border-x-0 rounded-none">
        <div className="max-w-4xl mx-auto px-4 sm:px-6">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-3">
              <button onClick={() => router.push('/chat')} className="p-2 rounded-lg hover:bg-white/5 text-curo-text-dim hover:text-white transition-colors">
                <ArrowLeft size={18} />
              </button>
              <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-curo-accent to-curo-teal flex items-center justify-center">
                <Brain size={18} className="text-white" />
              </div>
              <div>
                <h1 className="text-sm font-bold text-curo-text flex items-center gap-2">
                  Curo Assistant
                  <span className="text-[10px] text-curo-accent px-1.5 py-0.5 rounded bg-curo-accent/10 border border-curo-accent/20">AGENTIC</span>
                </h1>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <div className="w-1.5 h-1.5 rounded-full bg-curo-teal animate-pulse" />
                  <span className="text-[10px] text-curo-text-dim uppercase tracking-[0.1em]">
                    {triageFinished ? 'Interview Complete' : 'Clinical Interview in Progress'}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              {/* Progress */}
              <div className="hidden sm:flex flex-col items-end">
                <span className="text-[10px] uppercase font-bold text-curo-text-dim tracking-tight">Progress</span>
                <div className="flex items-center gap-2 mt-0.5">
                  <div className="w-24 h-1.5 rounded-full bg-curo-border overflow-hidden">
                    <div
                      className="h-full bg-curo-accent transition-all duration-500 ease-out rounded-full"
                      style={{ width: `${Math.min(100, (questionCount / 5) * 100)}%` }}
                    />
                  </div>
                  <span className="text-[10px] font-bold text-curo-accent">{Math.min(100, Math.round((questionCount / 5) * 100))}%</span>
                </div>
              </div>
              <button onClick={handleRestart} className="p-2 rounded-lg hover:bg-white/5 text-curo-text-dim hover:text-curo-accent transition-colors" title="Restart">
                <RefreshCw size={16} />
              </button>
              <div className="w-px h-5 bg-curo-border hidden sm:block mx-1"></div>
              <button onClick={() => signOut(auth)} className="p-2 rounded-lg hover:bg-curo-rose/10 text-curo-text-dim hover:text-curo-rose transition-colors" title="Sign Out">
                <LogOut size={16} />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Chat Area */}
      <main className="flex-1 max-w-4xl mx-auto w-full px-4 sm:px-6 py-6">
        <div className="space-y-5">
          {triageMessages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}>
              {msg.role === 'assistant' && (
                <div className="w-8 h-8 rounded-full bg-curo-accent/20 flex items-center justify-center mr-3 shrink-0 mt-1">
                  <Brain size={16} className="text-curo-accent" />
                </div>
              )}
              <div className={`${msg.role === 'user' ? 'max-w-[70%]' : 'max-w-[80%]'} p-4 rounded-2xl ${
                msg.role === 'user'
                  ? 'bg-curo-accent text-white rounded-tr-sm shadow-lg shadow-curo-accent/20'
                  : 'bg-white/[0.05] border border-curo-border text-curo-text rounded-tl-sm'
              }`}>
                <p className={msg.role === 'assistant' ? 'text-[15px] leading-relaxed' : 'text-sm'}>{msg.content}</p>
              </div>
            </div>
          ))}

          {triageLoading && (
            <div className="flex justify-start animate-fade-in">
              <div className="w-8 h-8 rounded-full bg-curo-accent/20 flex items-center justify-center mr-3 shrink-0">
                <Brain size={16} className="text-curo-accent" />
              </div>
              <div className="bg-white/[0.05] border border-curo-border p-4 rounded-2xl rounded-tl-sm">
                <div className="flex gap-1.5">
                  <div className="w-2 h-2 bg-curo-accent/60 rounded-full animate-bounce [animation-delay:-0.3s]" />
                  <div className="w-2 h-2 bg-curo-accent/60 rounded-full animate-bounce [animation-delay:-0.15s]" />
                  <div className="w-2 h-2 bg-curo-accent/60 rounded-full animate-bounce" />
                </div>
              </div>
            </div>
          )}
          <div ref={chatBottomRef} />
        </div>
      </main>

      {/* Bottom Action Area */}
      <div className="sticky bottom-0 glass-card-strong border-b-0 border-x-0 rounded-none">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 py-5">
          {triageFinished ? (
            <div className="flex flex-col animate-fade-in gap-3">
              <p className="text-sm text-center text-curo-text-dim mb-1">How would you like to analyze this summary?</p>
              <div className="flex flex-col sm:flex-row items-center gap-3">
                <button onClick={() => handleAnalyze('generic')} className="flex-1 w-full flex items-center justify-center gap-2.5 py-3.5 rounded-xl border border-curo-accent/30 bg-curo-accent/10 text-curo-accent hover:bg-curo-accent hover:text-white transition-all shadow-lg shadow-curo-accent/5">
                  <Globe size={18} />
                  Generic Web Research
                </button>
                <button onClick={() => handleAnalyze('deep-research')} className="glow-btn flex-1 w-full flex items-center justify-center gap-2.5 py-3.5">
                  <Database size={18} />
                  Deep Vault Research
                </button>
                <button onClick={handleRestart} className="flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl border border-curo-border bg-white/[0.02] text-curo-text-muted hover:text-curo-text hover:bg-white/[0.05] transition-all w-full sm:w-auto mt-2 sm:mt-0">
                  <RefreshCw size={16} />
                </button>
              </div>
            </div>
          ) : !triageLoading && triageMessages.length > 0 ? (
            <div className="flex flex-col gap-4 animate-fade-in w-full">
              {/* Options Row (if any, excluding 'Other') */}
              {triageMessages[triageMessages.length - 1].options && (
                <div className="flex flex-wrap gap-2.5 justify-center">
                  {triageMessages[triageMessages.length - 1].options
                    ?.filter(opt => !opt.toLowerCase().includes('other (please specify)'))
                    .map((opt, i) => (
                      <button
                        key={i}
                        onClick={() => handleTriageSelect(opt)}
                        className={`px-5 py-2.5 rounded-full border text-sm font-medium transition-all hover:scale-105 active:scale-95 border-curo-accent/30 bg-curo-accent/5 hover:bg-curo-accent hover:text-white text-curo-accent hover:shadow-lg hover:shadow-curo-accent/10`}
                      >
                        {opt}
                      </button>
                    ))}
                </div>
              )}
              
              {/* Always-visible text input form */}
              <form onSubmit={handleOtherSubmit} className="flex gap-3 w-full">
                <input
                  autoFocus
                  type="text"
                  value={triageInput}
                  onChange={(e) => setTriageInput(e.target.value)}
                  placeholder="Type your own response here..."
                  className="flex-1 curo-input text-sm px-4 h-12"
                />
                <button
                  type="submit"
                  disabled={!triageInput.trim() || triageLoading}
                  className="w-12 h-12 rounded-xl bg-curo-accent flex items-center justify-center text-white disabled:opacity-50 transition-all shadow-lg shadow-curo-accent/20 hover:scale-105 active:scale-95"
                >
                  <Send size={18} />
                </button>
              </form>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3 animate-pulse">
              <p className="text-xs text-curo-text-dim font-bold uppercase tracking-widest">
                {questionCount >= 5 ? "Preparing clinical summary..." : "Thinking..."}
              </p>
              <div className="w-48 h-1 rounded-full bg-curo-border overflow-hidden">
                <div className="h-full bg-curo-teal animate-loading-bar" />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
