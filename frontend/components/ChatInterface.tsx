'use client';

import { useState, useRef, useEffect } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import ResultsDashboard from './ResultsDashboard';
import HistorySidebar from './HistorySidebar';
import { Send, Activity, Sparkles, Brain, Search, BookOpen, FlaskConical, History, LogOut, Mic, MicOff, ChevronDown, ChevronUp, UserSquare2, RefreshCw, FolderOpen, Database, Plus, Paperclip, AlertCircle, CheckCircle2, ChevronRight, File, Pill, ExternalLink, Globe } from 'lucide-react';
import VoiceVisualizer from './VoiceVisualizer';
import { auth, db } from '../lib/firebase';
import { collection, addDoc, serverTimestamp, doc, updateDoc } from 'firebase/firestore';
import { signOut } from 'firebase/auth';

interface AnalysisResult {
  response: string;
  extracted_ddx: string[];
  winning_diagnosis: string;
  abstracts: Array<{ title: string; abstract: string; url?: string }>;
  graph_data: {
    nodes: Array<{ id: string; label: string; fx?: number; fy?: number }>;
    links: Array<{ source: string; target: string; label: string }>;
  };
}

type ChatMessage = {
  id: string;
  role: 'user' | 'assistant';
  content?: string;
  mode?: 'generic' | 'deep-research';
  result?: AnalysisResult;
  sources?: Array<{ title: string; url: string; snippet: string }>;
  isLoading?: boolean;
  error?: string;
};

const GENERIC_PIPELINE_STEPS = [
  { label: 'Launching multi-angle web search…', duration: 2000 },
  { label: 'Scraping medical sources & articles…', duration: 5000 },
  { label: 'Analyzing treatment guidelines…', duration: 4000 },
  { label: 'Cross-referencing clinical data…', duration: 4000 },
  { label: 'Synthesizing researched response…', duration: 5000 },
];

const PIPELINE_STEPS = [
  { label: 'Extracting clinical entities…', duration: 4000 },
  { label: 'Retrieving medical literature…', duration: 8000 },
  { label: 'Building knowledge graph…', duration: 6000 },
  { label: 'Synthesizing clinical analysis…', duration: 12000 },
];

function renderInlineText(text: string, keyPrefix: string = '') {
  // Split on bold markers, italic markers, and citation markers like [1], [2], etc.
  const parts = text.split(/(\*\*.*?\*\*|\*[^*\n]+?\*|\[\d+\])/g);
  return parts.map((part, j) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return (
        <strong key={`${keyPrefix}-b${j}`} className="text-curo-text font-semibold">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith('*') && part.endsWith('*') && part.length > 2 && !part.startsWith('**')) {
      return (
        <em key={`${keyPrefix}-i${j}`} className="text-curo-text-muted italic text-[13px]">
          {part.slice(1, -1)}
        </em>
      );
    }
    const citationMatch = part.match(/^\[(\d+)\]$/);
    if (citationMatch) {
      const num = citationMatch[1];
      return (
        <a
          key={`${keyPrefix}-cite-${j}`}
          href={`#source-${num}`}
          onClick={(e) => {
            e.preventDefault();
            document.getElementById(`source-${num}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
            const el = document.getElementById(`source-${num}`);
            if (el) {
              el.classList.add('ring-2', 'ring-curo-accent', 'scale-[1.02]');
              setTimeout(() => el.classList.remove('ring-2', 'ring-curo-accent', 'scale-[1.02]'), 1500);
            }
          }}
          className="inline-flex items-center justify-center min-w-[18px] h-[18px] px-1 mx-0.5 text-[10px] font-bold bg-curo-teal/20 text-curo-teal rounded-md hover:bg-curo-teal/30 transition-colors cursor-pointer align-super leading-none no-underline"
          title={`View Source ${num}`}
        >
          {num}
        </a>
      );
    }
    return <span key={`${keyPrefix}-${j}`}>{part}</span>;
  });
}

function parseResponse(text: string) {
  if (!text) return null;

  const lines = text.split('\n');
  const elements: React.ReactNode[] = [];
  let lineIdx = 0;
  let elementKey = 0;

  while (lineIdx < lines.length) {
    const trimmed = lines[lineIdx].trim();

    // Skip empty lines
    if (!trimmed) { lineIdx++; continue; }

    // --- Markdown ## headers ---
    const h2Match = trimmed.match(/^#{1,3}\s+(.+)$/);
    if (h2Match) {
      elements.push(
        <div key={`sec-${elementKey++}`} className="flex items-center gap-2.5 mt-7 mb-3 first:mt-1">
          <div className="w-1 h-5 rounded-full bg-gradient-to-b from-curo-accent to-curo-teal shrink-0" />
          <h3 className="text-[15px] font-bold text-curo-text tracking-tight">{renderInlineText(h2Match[1], `h-${lineIdx}`)}</h3>
        </div>
      );
      lineIdx++;
      continue;
    }

    // --- Section headers: lines that are ONLY bold text like **Header** or **Header**: ---
    const headerMatch = trimmed.match(/^\*\*([^*]+)\*\*[:\s\-\u2013\u2014]*$/);
    if (headerMatch && trimmed.length < 100) {
      elements.push(
        <div key={`sec-${elementKey++}`} className="flex items-center gap-2.5 mt-7 mb-3 first:mt-1">
          <div className="w-1 h-5 rounded-full bg-gradient-to-b from-curo-accent to-curo-teal shrink-0" />
          <h3 className="text-[15px] font-bold text-curo-text tracking-tight">{headerMatch[1].trim()}</h3>
        </div>
      );
      lineIdx++;
      continue;
    }

    // --- Bullet lists (-, \u2022, or * but not **) ---
    if (/^[-\u2022]\s/.test(trimmed) || (/^\*\s/.test(trimmed) && !/^\*\*/.test(trimmed))) {
      const items: string[] = [];
      while (lineIdx < lines.length) {
        const l = lines[lineIdx].trim();
        if (!l) break;
        if (/^[-\u2022]\s/.test(l) || (/^\*\s/.test(l) && !/^\*\*/.test(l))) {
          items.push(l.replace(/^[-\u2022*]\s*/, ''));
          lineIdx++;
        } else {
          break;
        }
      }
      if (items.length > 0) {
        elements.push(
          <ul key={`ul-${elementKey++}`} className="space-y-2.5 mb-5 ml-0.5">
            {items.map((item, j) => (
              <li key={j} className="flex items-start gap-2.5 text-[14px] text-curo-text/85 leading-relaxed">
                <span className="mt-[9px] w-1.5 h-1.5 rounded-full bg-curo-teal/50 shrink-0" />
                <span className="flex-1">{renderInlineText(item, `uli-${elementKey}-${j}`)}</span>
              </li>
            ))}
          </ul>
        );
      }
      continue;
    }

    // --- Numbered lists (1. or 1) style) ---
    if (/^\d+[.)]\s/.test(trimmed)) {
      const items: string[] = [];
      while (lineIdx < lines.length) {
        const l = lines[lineIdx].trim();
        if (!l) break;
        if (/^\d+[.)]\s/.test(l)) {
          items.push(l.replace(/^\d+[.)]\s*/, ''));
          lineIdx++;
        } else {
          break;
        }
      }
      if (items.length > 0) {
        elements.push(
          <ol key={`ol-${elementKey++}`} className="space-y-2.5 mb-5 ml-0.5">
            {items.map((item, j) => (
              <li key={j} className="flex items-start gap-3 text-[14px] text-curo-text/85 leading-relaxed">
                <span className="mt-0.5 min-w-[22px] h-[22px] rounded-full bg-curo-accent/10 flex items-center justify-center text-[11px] font-bold text-curo-accent shrink-0">{j + 1}</span>
                <span className="flex-1">{renderInlineText(item, `oli-${elementKey}-${j}`)}</span>
              </li>
            ))}
          </ol>
        );
      }
      continue;
    }

    // --- Regular paragraph: collect consecutive text lines ---
    const paraLines: string[] = [];
    while (lineIdx < lines.length) {
      const l = lines[lineIdx].trim();
      if (!l) { lineIdx++; break; }
      if (/^[-\u2022]\s/.test(l)) break;
      if (/^\*\s/.test(l) && !/^\*\*/.test(l)) break;
      if (/^\d+[.)]\s/.test(l)) break;
      if (/^#{1,3}\s/.test(l)) break;
      if (l.match(/^\*\*[^*]+\*\*[:\s\-\u2013\u2014]*$/) && l.length < 100) break;
      paraLines.push(l);
      lineIdx++;
    }

    if (paraLines.length > 0) {
      const paraText = paraLines.join(' ');

      // Detect disclaimer / italic wrapper
      const stripped = paraText.trim();
      if (/^\*[^*]/.test(stripped) && /[^*]\*$/.test(stripped)) {
        elements.push(
          <p key={`disc-${elementKey++}`} className="mt-6 pt-4 border-t border-white/[0.06] text-[12px] text-curo-text-dim/70 italic leading-relaxed">
            {stripped.replace(/^\*+\s*|\s*\*+$/g, '')}
          </p>
        );
      } else {
        elements.push(
          <p key={`p-${elementKey++}`} className="mb-3.5 leading-[1.85] text-[14px] text-curo-text/85">
            {renderInlineText(paraText, `p-${elementKey}`)}
          </p>
        );
      }
    }
  }

  return elements.length > 0 ? elements : null;
}

export default function ChatInterface() {
  const router = useRouter();
  const searchParams = useSearchParams();
  
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<'generic' | 'deep-research'>('generic');
  const [isModeDropdownOpen, setIsModeDropdownOpen] = useState(false);
  
  const [loading, setLoading] = useState(false);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [pipelineStep, setPipelineStep] = useState(0);
  
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isVitalsOpen, setIsVitalsOpen] = useState(false);
  
  const [age, setAge] = useState('');
  const [sex, setSex] = useState('Not specified');
  const [heartRate, setHeartRate] = useState('');
  const [bloodPressure, setBloodPressure] = useState('');
  const [spo2, setSpo2] = useState('');
  const [temp, setTemp] = useState('');
  const [respRate, setRespRate] = useState('');

  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [audioStream, setAudioStream] = useState<MediaStream | null>(null);
  
  const [recordsCount, setRecordsCount] = useState(0);
  const [autoRunDone, setAutoRunDone] = useState(false);
  
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const pipelineInterval = useRef<NodeJS.Timeout | null>(null);
  
  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const audioChunksRef = useRef<Float32Array[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const fetchRecordsCount = async () => {
      if (!auth.currentUser) return;
      try {
        const res = await fetch(`http://127.0.0.1:8000/api/records/list?user_id=${encodeURIComponent(auth.currentUser.uid)}`);
        if (res.ok) {
          const data = await res.json();
          setRecordsCount(data.total_chunks || 0);
        }
      } catch { /* silent */ }
    };
    fetchRecordsCount();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [query]);

  // Initial redirect query parsing (Triage assistant)
  useEffect(() => {
    const q = searchParams.get('q');
    if (q && !autoRunDone) {
      setQuery(q);
      setAutoRunDone(true);
      setTimeout(() => handleSendMessage(undefined, q), 500);
    }
  }, [searchParams, autoRunDone]);

  useEffect(() => {
    if (loading) {
      setPipelineStep(0);
      let step = 0;
      const steps = mode === 'generic' ? GENERIC_PIPELINE_STEPS : PIPELINE_STEPS;
      const advance = () => {
        step++;
        if (step < steps.length) {
          setPipelineStep(step);
          pipelineInterval.current = setTimeout(advance, steps[step].duration);
        }
      };
      pipelineInterval.current = setTimeout(advance, steps[0].duration);
    } else {
      if (pipelineInterval.current) clearTimeout(pipelineInterval.current);
    }
    return () => {
      if (pipelineInterval.current) clearTimeout(pipelineInterval.current);
    };
  }, [loading, mode]);

  const encodeWAV = (samples: Float32Array, sampleRate: number) => {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const writeString = (offset: number, string: string) => {
      for (let i = 0; i < string.length; i++) view.setUint8(offset + i, string.charCodeAt(i));
    };
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, samples.length * 2, true);
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
    return new Blob([view], { type: 'audio/wav' });
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setAudioStream(stream);
      setIsRecording(true);
      
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      audioContextRef.current = audioContext;
      
      await audioContext.audioWorklet.addModule('/audio-processor.js');
      
      const source = audioContext.createMediaStreamSource(stream);
      const workletNode = new AudioWorkletNode(audioContext, 'audio-processor');
      workletNodeRef.current = workletNode;
      
      audioChunksRef.current = [];
      
      workletNode.port.onmessage = (e) => {
        audioChunksRef.current.push(new Float32Array(e.data));
      };
      
      source.connect(workletNode);
      workletNode.connect(audioContext.destination);
    } catch (err) {
      console.error("Recording error:", err);
    }
  };

  const stopRecording = async () => {
    if (!isRecording) return;
    setIsRecording(false);
    setIsTranscribing(true);
    
    if (audioStream) {
      audioStream.getTracks().forEach(track => track.stop());
      setAudioStream(null);
    }
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
    }

    const totalLength = audioChunksRef.current.reduce((acc, chunk) => acc + chunk.length, 0);
    const mergedSamples = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of audioChunksRef.current) {
      mergedSamples.set(chunk, offset);
      offset += chunk.length;
    }

    if (totalLength === 0) {
      setIsTranscribing(false);
      return;
    }

    const wavBlob = encodeWAV(mergedSamples, 16000);
    try {
      const formData = new FormData();
      formData.append('file', wavBlob, 'audio.wav');
      const response = await fetch('http://localhost:8000/api/asr', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('ASR server error');
      const data = await response.json();
      if (data.transcript) {
        setQuery(prev => (prev.trim() + ' ' + data.transcript).trim());
      }
    } catch (err) {
       // Silent fail or toast error
    } finally {
      setIsTranscribing(false);
    }
  };

  const toggleRecording = () => {
    if (isRecording) stopRecording();
    else startRecording();
  };
  
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0 || !auth.currentUser) return;
    
    const msgId = Date.now().toString();
    setMessages(prev => [...prev, {
      id: msgId,
      role: 'user',
      content: `Uploading ${files.length} document(s)...`,
      isLoading: true
    }]);
    
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }
    
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/records/upload?user_id=${auth.currentUser.uid}`, {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        const data = await res.json();
        setRecordsCount(prev => prev + data.total_chunks);
        setMessages(prev => prev.filter(m => m.id !== msgId).concat([{
          id: Date.now().toString(),
          role: 'assistant',
          content: `✅ Successfully uploaded and indexed ${files.length} document(s) to your health records. You can now reference them in your queries.`,
          mode: 'generic'
        }]));
      } else {
        throw new Error('Upload failed');
      }
    } catch (err) {
      setMessages(prev => prev.filter(m => m.id !== msgId).concat([{
        id: Date.now().toString(),
        role: 'assistant',
        content: `❌ Failed to upload documents. Please try again.`,
        mode: 'generic'
      }]));
    }
  };

  const handleSendMessage = async (e?: React.FormEvent, overrideQuery?: string) => {
    if (e) e.preventDefault();
    const targetQuery = (overrideQuery || query).trim();
    if (!targetQuery || loading || isTranscribing) return;

    setQuery('');
    setLoading(true);

    const newMsgId = Date.now().toString();
    const loadingMsgId = (Date.now() + 1).toString();
    
    const newMessages: ChatMessage[] = [
      ...messages,
      { id: newMsgId, role: 'user', content: targetQuery }
    ];
    setMessages([...newMessages, { id: loadingMsgId, role: 'assistant', isLoading: true, mode }]);

    try {
      let endpoint = '';
      let payload: any = {};
      let isFollowUp = false;

      if (mode === 'generic') {
        endpoint = 'http://127.0.0.1:8000/api/generic-analyze';
        payload = { query: targetQuery, user_id: auth.currentUser?.uid };
      } else {
        const prevResults = newMessages.filter(m => m.mode === 'deep-research' && m.result);
        const lastResultMsg = prevResults[prevResults.length - 1];
        
        if (lastResultMsg?.result) {
          isFollowUp = true;
          endpoint = 'http://127.0.0.1:8000/api/chat';
          const chatHistory = newMessages.map(m => ({ 
            role: m.role, 
            content: m.content || (m.result ? `[Deep Research Results for: ${m.result.winning_diagnosis}]` : '') 
          }));
          payload = {
            message: targetQuery,
            history: chatHistory,
            context: lastResultMsg.result.response || '',
            user_id: auth.currentUser?.uid
          };
        } else {
          endpoint = 'http://127.0.0.1:8000/api/analyze';
          payload = {
            query: targetQuery,
            demography: { age, sex: sex !== 'Not specified' ? sex : undefined, heartRate, bloodPressure, spo2, temp, respRate },
            user_id: auth.currentUser?.uid
          };
        }
      }

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`Analysis failed (${response.status})`);
      }

      const data = await response.json();
      let finalAssistantMsg: ChatMessage;

      if (mode === 'generic') {
         finalAssistantMsg = {
            id: loadingMsgId,
            role: 'assistant',
            content: data.response,
            mode: 'generic',
            sources: data.sources
         };
      } else {
         if (isFollowUp) {
            finalAssistantMsg = {
               id: loadingMsgId,
               role: 'assistant',
               content: data.response,
               mode: 'deep-research'
            };
         } else {
            finalAssistantMsg = {
               id: loadingMsgId,
               role: 'assistant',
               content: data.response,
               mode: 'deep-research',
               result: data
            };
            
            // Save initial session to Firebase
            if (auth.currentUser && data.winning_diagnosis) {
                try {
                const docRef = await addDoc(collection(db, 'sessions'), {
                    userId: auth.currentUser.uid,
                    query: targetQuery,
                    winning_diagnosis: data.winning_diagnosis,
                    result_data: data,
                    messages: [],
                    timestamp: serverTimestamp()
                });
                setCurrentSessionId(docRef.id);
                } catch (fbErr) {
                console.error('Failed to save session to Firestore', fbErr);
                }
            }
         }
      }

      setMessages(prev => prev.map(m => m.id === loadingMsgId ? finalAssistantMsg : m));

      // Update session if it's a follow-up
      if (isFollowUp && currentSessionId) {
          try {
             // We reconstruct the basic chat format for FB to store follow-ups
             const fbMessages = newMessages
                .filter(m => m.content && !m.result)
                .concat(finalAssistantMsg)
                .map(m => ({ role: m.role, content: m.content }));
             
             await updateDoc(doc(db, 'sessions', currentSessionId), {
                messages: fbMessages
             });
          } catch (e) { console.error('Failed to update FB session', e); }
      }

    } catch (err: any) {
      console.error(err);
      setMessages(prev => prev.map(m => m.id === loadingMsgId ? {
        ...m,
        isLoading: false,
        error: err.message || 'An unexpected error occurred. Please try again.'
      } : m));
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-curo-bg overflow-hidden relative">
      {/* Header */}
      <header className="flex-shrink-0 z-20 glass-card-strong border-t-0 border-x-0 rounded-none h-14 flex items-center justify-between px-4 sm:px-8">
         <div className="flex items-center gap-3 cursor-pointer hover:opacity-80 transition-opacity" onClick={() => router.push('/')}>
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-curo-accent to-curo-teal flex items-center justify-center">
              <Activity size={16} className="text-white" />
            </div>
            <h1 className="text-base font-bold gradient-text">CURO AI</h1>
         </div>
         <div className="flex items-center gap-3">
            <button onClick={() => router.push('/records')} className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg border border-curo-teal/20 bg-curo-teal/5 hover:bg-curo-teal/10 text-xs text-curo-teal transition-colors">
              <FolderOpen size={14} /> Records
              {recordsCount > 0 && <span className="bg-curo-teal/20 text-curo-teal px-1.5 py-0.5 rounded-full text-[10px]">{recordsCount} chunks</span>}
            </button>
            <button onClick={() => router.push('/triage')} className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg border border-curo-accent/20 bg-curo-accent/5 hover:bg-curo-accent/10 text-xs text-curo-accent transition-colors">
              <Brain size={14} /> Curo Assistant
            </button>
            <button onClick={() => setIsSidebarOpen(true)} className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-curo-border bg-white/[0.02] hover:bg-white/[0.05] text-xs text-curo-text-muted hover:text-curo-text transition-colors">
              <History size={14} /> History
            </button>
            <button onClick={() => auth.signOut()} className="flex items-center justify-center w-8 h-8 rounded-lg bg-curo-rose/10 text-curo-rose hover:bg-curo-rose/20 transition-colors">
              <LogOut size={14} />
            </button>
         </div>
      </header>

      {/* Main Chat Area */}
      <div className="flex-1 overflow-y-auto w-full scrollbar-none">
         <div className="max-w-4xl mx-auto px-4 py-8 pb-48">
            {messages.length === 0 ? (
               <div className="flex flex-col items-center justify-center h-[60vh] text-center px-4 animate-fade-in">
                  <div className="w-16 h-16 rounded-3xl bg-curo-bg border border-curo-border flex items-center justify-center mb-6 shadow-glow relative overflow-hidden group">
                     <div className="absolute inset-0 bg-gradient-to-br from-curo-accent/20 to-curo-teal/20 opacity-0 group-hover:opacity-100 transition-opacity" />
                     <Sparkles size={28} className="text-curo-accent relative z-10" />
                  </div>
                  <h2 className="text-3xl font-bold text-curo-text mb-3">How can Curo AI help today?</h2>
                  <p className="text-curo-text-muted text-base max-w-md mb-8">
                     Quickly look up clinical treatments or switch to deep research for an extensive medical literature and graph analysis.
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full max-w-lg mx-auto">
                     <button onClick={() => { setMode('generic'); setQuery('What are the best home remedies for a severe headache?'); }} className="p-4 rounded-2xl border border-curo-border bg-white/[0.02] hover:bg-white/[0.04] text-left transition-colors">
                        <Pill size={16} className="text-curo-teal mb-2" />
                        <p className="text-sm text-curo-text font-medium">Home remedies for headache</p>
                        <p className="text-xs text-curo-text-dim mt-1">Generic Mode lookup</p>
                     </button>
                     <button onClick={() => { setMode('deep-research'); setQuery('A 45yo male presents with acute chest pain radiating to the jaw...'); }} className="p-4 rounded-2xl border border-curo-border bg-white/[0.02] hover:bg-white/[0.04] text-left transition-colors">
                        <Brain size={16} className="text-curo-purple mb-2" />
                        <p className="text-sm text-curo-text font-medium">Analyze complex presentation</p>
                        <p className="text-xs text-curo-text-dim mt-1">Deep Research pipeline</p>
                     </button>
                  </div>
               </div>
            ) : (
               <div className="space-y-8">
                  {messages.map((msg, idx) => (
                     <div key={msg.id} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : ''} animate-fade-in`}>
                        {msg.role === 'assistant' && (
                           <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-curo-accent to-curo-teal flex items-center justify-center shrink-0 shadow-lg mt-1 relative overflow-hidden">
                              <Activity size={14} className="text-white relative z-10" />
                           </div>
                        )}
                        <div className={`max-w-[85%] sm:max-w-full ${msg.role === 'user' ? 'bg-white/[0.04] border border-curo-border rounded-2xl px-5 py-4 inline-block' : 'flex-1'}`}>
                           
                           {/* User Message */}
                           {msg.role === 'user' && (
                              <p className="text-curo-text text-[15px] leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                           )}

                           {/* Assistant Loading State */}
                           {msg.role === 'assistant' && msg.isLoading && (
                              <div className="pt-2">
                                 <div className="space-y-3 px-5 py-4 bg-white/[0.02] border border-curo-border rounded-xl w-full max-w-sm">
                                    {(msg.mode === 'generic' ? GENERIC_PIPELINE_STEPS : PIPELINE_STEPS).map((step, i) => (
                                       <div key={i} className={`flex items-center gap-3 text-sm ${i === pipelineStep ? 'text-curo-accent font-medium' : i < pipelineStep ? 'text-curo-teal' : 'text-curo-text-dim opacity-50'}`}>
                                          {i < pipelineStep ? <CheckCircle2 size={14} className="shrink-0"/> : i === pipelineStep ? <div className="w-3.5 h-3.5 rounded-full border-2 border-curo-accent border-t-transparent animate-spin shrink-0"/> : <div className="w-3.5 h-3.5 rounded-full border-2 border-curo-border shrink-0"/>}
                                          {step.label}
                                       </div>
                                    ))}
                                 </div>
                              </div>
                           )}

                           {/* Assistant Error */}
                           {msg.role === 'assistant' && msg.error && (
                              <div className="flex items-start gap-3 p-4 rounded-xl bg-curo-rose/10 border border-curo-rose/30 text-curo-rose text-sm">
                                 <AlertCircle size={16} className="mt-0.5 shrink-0" />
                                 <p>{msg.error}</p>
                              </div>
                           )}

                           {/* Assistant Content */}
                           {msg.role === 'assistant' && !msg.isLoading && !msg.error && (
                              <div className="space-y-4 pt-1 w-full">
                                 {msg.content && !msg.result && (
                                    <div className="max-w-none">
                                       {parseResponse(msg.content)}
                                    </div>
                                 )}

{/* Sources for Generic Mode — Adaptive Reference Grid */}
                                  {msg.mode === 'generic' && msg.sources && msg.sources.length > 0 && (() => {
                                     const validSources = msg.sources.filter(s => !!s.url);
                                     const count = validSources.length;
                                     const gridClass = count === 1
                                        ? 'grid-cols-1'
                                        : count === 2
                                           ? 'grid-cols-1 sm:grid-cols-2'
                                           : count <= 4
                                              ? 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-2'
                                              : 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3';
                                     return count > 0 ? (
                                        <div className="mt-8 pt-6 border-t border-white/[0.06]">
                                           <div className="flex items-center gap-2.5 mb-5">
                                              <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-curo-teal/20 to-curo-accent/10 flex items-center justify-center">
                                                 <Globe size={14} className="text-curo-teal" />
                                              </div>
                                              <p className="text-[11px] font-semibold text-curo-text-muted uppercase tracking-widest">
                                                 References · {count} source{count !== 1 ? 's' : ''} cited
                                              </p>
                                           </div>
                                           <div className={`grid ${gridClass} gap-3`}>
                                              {validSources.map((src, i) => {
                                                 const sourceNum = (src as any).index || (i + 1);
                                                 let hostname = '';
                                                 try { hostname = new URL(src.url).hostname.replace('www.', ''); } catch { hostname = src.url.slice(0, 30); }
                                                 return (
                                                    <a
                                                       key={i}
                                                       id={`source-${sourceNum}`}
                                                       href={src.url}
                                                       target="_blank"
                                                       rel="noopener noreferrer"
                                                       onClick={(e) => {
                                                          e.preventDefault();
                                                          window.open(src.url, '_blank', 'noopener,noreferrer');
                                                       }}
                                                       className="group block px-4 py-3.5 bg-[#0a0f1a] border border-white/[0.06] rounded-xl hover:border-curo-teal/40 hover:bg-[#0f1729] transition-all duration-300 transform hover:scale-[1.02] hover:shadow-lg hover:shadow-curo-teal/5 cursor-pointer select-none"
                                                    >
                                                       <div className="flex items-start gap-3">
                                                          <div className="relative shrink-0">
                                                             <div className="w-8 h-8 rounded-lg bg-curo-teal/10 group-hover:bg-curo-teal/20 flex items-center justify-center transition-colors text-[11px] font-bold text-curo-teal border border-curo-teal/10 group-hover:border-curo-teal/30">
                                                                {sourceNum}
                                                             </div>
                                                          </div>
                                                          <div className="overflow-hidden flex-1 min-w-0">
                                                             <div className="flex items-center gap-2 mb-1.5">
                                                                <img
                                                                   src={`https://www.google.com/s2/favicons?domain=${hostname}&sz=16`}
                                                                   alt=""
                                                                   className="w-3.5 h-3.5 rounded-sm opacity-60 group-hover:opacity-100 transition-opacity"
                                                                   onError={(e) => { (e.target as HTMLImageElement).style.display = 'none'; }}
                                                                />
                                                                <span className="text-[10px] text-curo-teal/60 group-hover:text-curo-teal truncate transition-colors font-medium">{hostname}</span>
                                                                <ExternalLink size={10} className="text-curo-text-dim opacity-0 group-hover:opacity-80 transition-opacity ml-auto shrink-0" />
                                                             </div>
                                                             <p className="text-[12.5px] font-semibold text-curo-text/90 group-hover:text-curo-text leading-snug line-clamp-2 transition-colors">{src.title}</p>
                                                             {src.snippet && <p className="text-[11px] text-curo-text-dim/80 mt-1.5 line-clamp-2 leading-relaxed">{src.snippet}</p>}
                                                          </div>
                                                       </div>
                                                    </a>
                                                 );
                                              })}
                                           </div>
                                        </div>
                                     ) : null;
                                  })()}
                                 
                                 {/* Inline Results Dashboard for Deep Research */}
                                 {msg.mode === 'deep-research' && msg.result && (
                                    <div className="w-full mt-2 bg-[#0a0f18] border border-curo-border rounded-2xl overflow-hidden shadow-2xl pb-4">
                                       <ResultsDashboard 
                                          result={msg.result} 
                                          onBack={() => {}} 
                                          hideChat={true}
                                          isHistoryView={true} 
                                       />
                                    </div>
                                 )}
                              </div>
                           )}
                        </div>
                     </div>
                  ))}
                  <div ref={chatEndRef} className="h-4" />
               </div>
            )}
         </div>
      </div>

      {/* Sticky Bottom Input Area */}
      <div className="absolute bottom-0 left-0 right-0 z-30 bg-gradient-to-t from-[#060A14] via-[#060A14] to-transparent pt-12 pb-6 px-4">
         <div className="max-w-3xl mx-auto">
            {/* Vitals Panel (Only in deep research) */}
            {mode === 'deep-research' && (
               <div className="mb-3 ml-2">
                  <button 
                     onClick={() => setIsVitalsOpen(!isVitalsOpen)}
                     className="flex items-center gap-2 text-xs font-medium text-curo-text-dim hover:text-curo-accent transition-colors bg-[#0d1324] px-4 py-2 rounded-t-xl border border-b-0 border-curo-border"
                  >
                     <UserSquare2 size={14} /> Demographics & Vitals {isVitalsOpen ? <ChevronDown size={14}/> : <ChevronRight size={14}/>}
                  </button>
                  {isVitalsOpen && (
                     <div className="bg-[#0b0f19] border border-curo-border rounded-xl rounded-tl-none p-5 grid grid-cols-2 sm:grid-cols-4 gap-4 shadow-xl mb-4 animate-fade-in relative z-20">
                        <div>
                           <label className="block text-[10px] uppercase font-bold text-curo-text-dim mb-1">Age</label>
                           <input type="number" placeholder="e.g. 65" className="w-full bg-[#141b2a] border border-curo-border rounded-lg p-2 text-xs text-curo-text focus:border-curo-accent outline-none" value={age} onChange={e => setAge(e.target.value)} />
                        </div>
                        <div>
                           <label className="block text-[10px] uppercase font-bold text-curo-text-dim mb-1">Sex</label>
                           <select className="w-full bg-[#141b2a] border border-curo-border rounded-lg p-2 text-xs text-curo-text focus:border-curo-accent outline-none appearance-none" value={sex} onChange={e => setSex(e.target.value)}>
                              <option>Not specified</option>
                              <option>Male</option>
                              <option>Female</option>
                           </select>
                        </div>
                        <div>
                           <label className="block text-[10px] uppercase font-bold text-curo-text-dim mb-1">Heart (bpm)</label>
                           <input type="number" placeholder="e.g. 110" className="w-full bg-[#141b2a] border border-curo-border rounded-lg p-2 text-xs text-curo-text focus:border-curo-accent outline-none" value={heartRate} onChange={e => setHeartRate(e.target.value)} />
                        </div>
                        <div>
                           <label className="block text-[10px] uppercase font-bold text-curo-text-dim mb-1">BP (mmHg)</label>
                           <input type="text" placeholder="e.g. 140/90" className="w-full bg-[#141b2a] border border-curo-border rounded-lg p-2 text-xs text-curo-text focus:border-curo-accent outline-none" value={bloodPressure} onChange={e => setBloodPressure(e.target.value)} />
                        </div>
                     </div>
                  )}
               </div>
            )}

            {/* Input Container */}
            <div className="relative bg-[#0F1523] border border-curo-border rounded-[24px] shadow-2xl transition-all focus-within:border-curo-accent/50 focus-within:ring-2 focus-within:ring-curo-accent/20 overflow-visible z-30">
               
               {/* Mode Dropdown */}
               <div className="absolute -top-3.5 left-6 z-40">
                  <div className="relative">
                     <button 
                        onClick={() => setIsModeDropdownOpen(!isModeDropdownOpen)}
                        className="flex items-center gap-1.5 bg-[#141B2A] border border-curo-border hover:border-curo-accent/50 rounded-full px-4 py-1.5 shadow-md text-[13px] font-semibold text-curo-text transition-colors"
                     >
                        {mode === 'generic' ? <Sparkles size={14} className="text-curo-teal"/> : <Brain size={14} className="text-curo-purple"/>}
                        {mode === 'generic' ? 'Generic' : 'Deep Research'}
                        <ChevronDown size={14} className={`text-curo-text-dim transition-transform duration-200 ${isModeDropdownOpen ? 'rotate-180' : ''}`}/>
                     </button>
                     
                     {isModeDropdownOpen && (
                        <div className="absolute bottom-[calc(100%+8px)] left-0 w-56 bg-[#1A2333] border border-curo-border rounded-2xl shadow-2xl overflow-hidden animate-fade-in z-50">
                           <button onClick={() => { setMode('generic'); setIsModeDropdownOpen(false); }} className={`w-full text-left flex items-start gap-3 px-4 py-3.5 hover:bg-white/[0.04] transition-colors relative ${mode === 'generic' ? 'bg-white/[0.02]' : ''}`}>
                              {mode === 'generic' && <div className="absolute left-0 top-0 bottom-0 w-1 bg-curo-teal rounded-r-sm" />}
                              <div className="w-7 h-7 rounded-lg bg-curo-teal/10 flex items-center justify-center shrink-0 mt-0.5">
                                 <Sparkles size={14} className="text-curo-teal" />
                              </div>
                              <div>
                                 <div className={`text-sm font-semibold ${mode === 'generic' ? 'text-curo-text' : 'text-curo-text-muted'}`}>Generic</div>
                                 <div className="text-[11px] text-curo-text-dim mt-1 leading-snug">Web search & direct prescriptions for everyday symptoms.</div>
                              </div>
                           </button>
                           <button onClick={() => { setMode('deep-research'); setIsModeDropdownOpen(false); }} className={`w-full text-left flex items-start gap-3 px-4 py-3.5 hover:bg-white/[0.04] transition-colors border-t border-curo-border relative ${mode === 'deep-research' ? 'bg-white/[0.02]' : ''}`}>
                              {mode === 'deep-research' && <div className="absolute left-0 top-0 bottom-0 w-1 bg-curo-purple rounded-r-sm" />}
                              <div className="w-7 h-7 rounded-lg bg-curo-purple/10 flex items-center justify-center shrink-0 mt-0.5">
                                 <Brain size={14} className="text-curo-purple" />
                              </div>
                              <div>
                                 <div className={`text-sm font-semibold ${mode === 'deep-research' ? 'text-curo-text' : 'text-curo-text-muted'}`}>Deep Research</div>
                                 <div className="text-[11px] text-curo-text-dim mt-1 leading-snug">Full RAG pipeline, knowledge graphs, & clinical diagnosis.</div>
                              </div>
                           </button>
                        </div>
                     )}
                  </div>
               </div>

               <div className="flex items-end px-3 py-3 pt-7 gap-2 relative z-20">
                  <input type="file" ref={fileInputRef} className="hidden" multiple accept=".pdf" onChange={handleFileUpload} />
                  <button 
                     type="button" 
                     onClick={() => fileInputRef.current?.click()}
                     className="p-2.5 mb-0.5 flex-shrink-0 text-curo-text-dim hover:text-curo-text hover:bg-white/[0.05] rounded-xl transition-colors"
                     title="Upload clinical reports (PDF)"
                  >
                     <Plus size={22} />
                  </button>

                  <textarea
                     ref={textareaRef}
                     value={query}
                     onChange={(e) => setQuery(e.target.value)}
                     onKeyDown={handleKeyDown}
                     placeholder={isRecording ? "Listening..." : "Message Curo AI..."}
                     className="flex-1 max-h-48 min-h-[44px] bg-transparent text-curo-text text-[15px] placeholder-curo-text-dim resize-none border-none focus:outline-none focus:ring-0 px-2 py-3 scrollbar-none"
                     rows={1}
                     disabled={loading || isTranscribing}
                  />

                  {/* Mic Button */}
                  <button
                     type="button"
                     onClick={toggleRecording}
                     disabled={loading || isTranscribing}
                     className={`p-2.5 mb-0.5 flex-shrink-0 rounded-xl transition-colors ${isRecording ? 'bg-curo-rose/20 text-curo-rose animate-pulse' : 'text-curo-text-dim hover:text-curo-text hover:bg-white/[0.05]'}`}
                     title={isRecording ? "Stop dictation" : "Voice transcription"}
                  >
                     {isRecording ? <MicOff size={20} /> : <Mic size={20} />}
                  </button>

                  {/* Send Button */}
                  <button
                     type="button"
                     onClick={(e) => handleSendMessage(e)}
                     disabled={loading || !query.trim()}
                     className={`p-3 mb-0.5 flex-shrink-0 rounded-xl transition-all duration-200 ${query.trim() && !loading ? 'bg-white text-[#060A14] hover:bg-white/90 shadow-lg' : 'bg-white/[0.05] text-curo-text-dim cursor-not-allowed'}`}
                  >
                     {loading ? <RefreshCw size={16} className="animate-spin" /> : <Send size={16} className="ml-0.5" />}
                  </button>
               </div>
               
               {isRecording && (
                  <div className="absolute -top-12 left-0 right-0 z-10 flex justify-center">
                     <VoiceVisualizer isRecording={isRecording} stream={audioStream} />
                  </div>
               )}
            </div>
            
            <p className="text-center text-[10px] text-curo-text-dim mt-4">CURO AI is a clinical assistant. Not a substitute for professional medical advice.</p>
         </div>
      </div>

      <HistorySidebar 
         isOpen={isSidebarOpen} 
         onClose={() => setIsSidebarOpen(false)} 
         onSelectHistory={(data, pastQuery, id, pastMessages) => {
            // Reconstruct chat state from history
            if (data) {
               setMode('deep-research');
               setCurrentSessionId(id);
               // the history provides pastMessages for followups
               const reconstructed: ChatMessage[] = [];
               reconstructed.push({
                  id: 'hist-initial',
                  role: 'user',
                  content: pastQuery,
                  mode: 'deep-research'
               });
               
               reconstructed.push({
                  id: 'hist-initial-res',
                  role: 'assistant',
                  content: 'Loaded past session.',
                  mode: 'deep-research',
                  result: data
               });
               
               if (pastMessages && pastMessages.length) {
                  pastMessages.forEach((pm: any, i: number) => {
                     reconstructed.push({
                        id: `hist-followup-${i}`,
                        role: pm.role,
                        content: pm.content,
                        mode: 'deep-research'
                     });
                  });
               }
               setMessages(reconstructed);
            }
         }} 
      />
    </div>
  );
}