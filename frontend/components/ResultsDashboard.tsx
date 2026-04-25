'use client';

import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import dynamic from 'next/dynamic';
import {
  BookOpen,
  Network,
  Stethoscope,
  ChevronDown,
  ChevronUp,
  Shield,
  Award,
  FileText,
  MessageSquare,
  RefreshCw,
  Send,
  History,
  Search,
  ExternalLink,
  Download,
  Check,
} from 'lucide-react';
import { doc, updateDoc } from 'firebase/firestore';
import { db } from '../lib/firebase';

// Dynamically import force-graph wrapper to avoid SSR canvas issues and fix React Ref forwarding
const ForceGraph2D = dynamic(
  () => import('./NoSSRGraph'),
  {
    ssr: false,
    loading: () => (
      <div className="h-full flex items-center justify-center text-curo-text-dim text-sm">
        Initializing Graph Engine…
      </div>
    ),
  }
);

/* ── Types ──────────────────────────────── */
interface ResultsDashboardProps {
  result: {
    response: string;
    extracted_ddx: string[];
    winning_diagnosis: string;
    abstracts: Array<{ title: string; abstract: string; url?: string }>;
    graph_data: {
      nodes: Array<{ id: string; label: string; fx?: number; fy?: number }>;
      links: Array<{ source: string; target: string; label: string }>;
    };
  };
  sessionId?: string | null;
  initialMessages?: any[];
  userQuery?: string;
  userId?: string;
  onBack: () => void;
  onNodeClick?: (label: string) => void;
  isHistoryView?: boolean;
  hideChat?: boolean;
}

/* ── Typewriter hook ────────────────────── */
function useTypewriter(text: string, speed = 12) {
  const [displayed, setDisplayed] = useState('');
  const [done, setDone] = useState(false);

  useEffect(() => {
    setDisplayed('');
    setDone(false);
    let i = 0;
    const timer = setInterval(() => {
      i++;
      setDisplayed(text.slice(0, i));
      if (i >= text.length) {
        clearInterval(timer);
        setDone(true);
      }
    }, speed);
    return () => clearInterval(timer);
  }, [text, speed]);

  const skipToEnd = useCallback(() => {
    setDisplayed(text);
    setDone(true);
  }, [text]);

  return { displayed, done, skipToEnd };
}

/* ── Simple markdown-like parser ────────── */
function parseResponse(text: string) {
  // Split into paragraphs, render bold (**text**) and numbered lists
  const paragraphs = text.split(/\n{2,}/).filter(Boolean);
  return paragraphs.map((p, i) => {
    // Numbered list
    if (/^\d+\.\s/.test(p.trim())) {
      const items = p.split(/\n/).filter(Boolean);
      return (
        <ol key={i} className="list-decimal list-inside space-y-1.5 mb-4 text-curo-text/90 text-lg">
          {items.map((item, j) => (
            <li key={j} className="leading-relaxed">
              {item.replace(/^\d+\.\s*/, '')}
            </li>
          ))}
        </ol>
      );
    }
    // Bullet list
    if (/^[-•]\s/.test(p.trim())) {
      const items = p.split(/\n/).filter(Boolean);
      return (
        <ul key={i} className="list-disc list-inside space-y-1.5 mb-4 text-curo-text/90 text-lg">
          {items.map((item, j) => (
            <li key={j} className="leading-relaxed">
              {item.replace(/^[-•]\s*/, '')}
            </li>
          ))}
        </ul>
      );
    }
    // Bold markers
    const parts = p.split(/(\*\*.*?\*\*)/g);
    return (
      <p key={i} className="mb-4 leading-relaxed text-curo-text/90 text-lg">
        {parts.map((part, j) =>
          part.startsWith('**') && part.endsWith('**') ? (
            <strong key={j} className="text-curo-text font-semibold">
              {part.slice(2, -2)}
            </strong>
          ) : (
            part
          )
        )}
      </p>
    );
  });
}

/* ── Accordion ──────────────────────────── */
function AbstractAccordion({ title, abstract, url, index }: { title: string; abstract: string; url?: string; index: number }) {
  const [open, setOpen] = useState(false);

  return (
    <div
      className="glass-card overflow-hidden animate-slide-up"
      style={{ animationDelay: `${index * 0.1}s`, opacity: 0 }}
    >
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-start gap-3 p-4 sm:p-5 text-left hover:bg-white/[0.02] transition-colors"
      >
        <div className="w-7 h-7 rounded-md bg-curo-teal/15 flex items-center justify-center shrink-0 mt-0.5">
          <FileText size={14} className="text-curo-teal" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-lg font-bold text-curo-text leading-snug pr-6">{title}</p>
        </div>
        {open ? (
          <ChevronUp size={16} className="text-curo-text-dim shrink-0 mt-1" />
        ) : (
          <ChevronDown size={16} className="text-curo-text-dim shrink-0 mt-1" />
        )}
      </button>
      {open && (
        <div className="px-4 sm:px-5 pb-4 sm:pb-5 pt-0 animate-fade-in">
          <div className="pl-10">
            <div className="text-lg text-curo-text-muted leading-relaxed">{parseResponse(abstract)}</div>
            {url && (
              <a 
                href={url} 
                target="_blank" 
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 mt-3 text-xs font-medium text-curo-accent hover:text-curo-accent-light transition-colors"
              >
                Read Source <FileText size={12} />
              </a>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Node color helper ──────────────────── */
function getNodeColor(label: string, allLinks: Array<{ source: any; target: any; label: string }>, winningDiagnosis: string) {
  // Central topic / winning diagnosis gets a special color
  if (label === winningDiagnosis) return '#8b5cf6'; // purple for main diagnosis
  
  const l = label.toLowerCase();
  
  // Check if used as treatment
  const isTreatment = allLinks.some(
    (link) => {
      const rel = link.label?.toUpperCase() || '';
      return (rel.includes('TREAT') || rel.includes('PREVENT') || rel.includes('MANAGE')) &&
        ((typeof link.source === 'string' ? link.source : link.source?.id) === label ||
         (typeof link.target === 'string' ? link.target : link.target?.id) === label);
    }
  );
  if (isTreatment) return '#14b8a6'; // teal
  
  // Check if symptom
  const isSymptom = allLinks.some(
    (link) => {
      const rel = link.label?.toUpperCase() || '';
      return (rel.includes('PRESENTS') || rel.includes('CAUSE') || rel.includes('TRIGGER') || rel.includes('SYMPTOM')) &&
        (typeof link.target === 'string' ? link.target : link.target?.id) === label;
    }
  );
  if (isSymptom) return '#f59e0b'; // amber
  
  // Check if it's a BELONGS_TO relationship target (i.e., the topic)
  const isTopic = allLinks.some(
    (link) => link.label === 'BELONGS_TO' &&
      (typeof link.target === 'string' ? link.target : link.target?.id) === label
  );
  if (isTopic) return '#8b5cf6'; // purple
  
  // Check common treatment keywords
  if (l.includes('therapy') || l.includes('treatment') || l.includes('medication') || l.includes('drug') || l.includes('surgery') || l.includes('intervention')) {
    return '#14b8a6';
  }
  
  // Check common symptom keywords
  if (l.includes('pain') || l.includes('fever') || l.includes('nausea') || l.includes('fatigue') || l.includes('swelling')) {
    return '#f59e0b';
  }
  
  // Default = disease / entity
  return '#06b6d4'; // cyan
}

function getNodeSize(label: string, winningDiagnosis: string) {
  if (label === winningDiagnosis) return 14; // central topic is larger
  return 8;
}

/* ════════════════════════════════════════════
   MAIN COMPONENT
   ════════════════════════════════════════════ */
export default function ResultsDashboard({ result, sessionId, initialMessages, userQuery, userId, isHistoryView, hideChat, onBack, onNodeClick }: ResultsDashboardProps) {
  const { displayed, done, skipToEnd } = useTypewriter(result.response, 1);
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<any>(null);
  const [dims, setDims] = useState({ width: 0, height: 0 });

  // Automatically skip the typewriter animation if this is a pre-loaded history record
  useEffect(() => {
    if (isHistoryView) {
      skipToEnd();
    }
  }, [isHistoryView, skipToEnd]);

  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      setDims({
        width: entries[0].contentRect.width,
        height: entries[0].contentRect.height,
      });
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);
  const handleDownloadGraph = () => {
    const canvas = containerRef.current?.querySelector('canvas');
    if (!canvas) return;
    
    try {
      // Create a temporary canvas to draw the background and the graph
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = (canvas as HTMLCanvasElement).width;
      tempCanvas.height = (canvas as HTMLCanvasElement).height;
      const ctx = tempCanvas.getContext('2d');
      
      if (ctx) {
        // 1. Fill with the dark background color (#060a14)
        ctx.fillStyle = '#060a14';
        ctx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
        
        // 2. Draw the graph canvas on top
        ctx.drawImage(canvas as HTMLCanvasElement, 0, 0);
        
        // 3. Download the result
        const link = document.createElement('a');
        link.download = `curo-knowledge-graph-${result.winning_diagnosis.toLowerCase().replace(/\s+/g, '-')}.png`;
        link.href = tempCanvas.toDataURL('image/png');
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    } catch (err) {
      console.error("Failed to download graph:", err);
    }
  };


  // Follow-up Chat State
  const [messages, setMessages] = useState<Array<{role: 'user' | 'assistant', content: string}>>(initialMessages || []);
  const [newMessage, setNewMessage] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, chatLoading]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newMessage.trim() || chatLoading) return;

    const userMsg = newMessage.trim();
    const updatedWithUser = [...messages, { role: 'user' as const, content: userMsg }];
    setMessages(updatedWithUser);
    setNewMessage('');
    setChatLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMsg,
          history: messages,
          context: result.response,
          user_id: userId 
        }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => null);
        throw new Error(errData?.detail || 'Failed to fetch follow-up message');
      }
      
      const data = await response.json();
      const finalMessages = [
        ...updatedWithUser, 
        { role: 'assistant' as const, content: data.response }
      ];
      setMessages(finalMessages);

      // Sync to Firestore if session exists
      if (sessionId) {
        try {
          await updateDoc(doc(db, 'sessions', sessionId), {
            messages: finalMessages
          });
        } catch (syncErr) {
          console.error("Failed to sync chat to Firestore:", syncErr);
        }
      }
    } catch (err: any) {
      console.error(err);
      setMessages(prev => [...prev, { role: 'assistant', content: `Chat error: ${err.message || 'Connection failed. Please try again.'}` }]);
    } finally {
      setChatLoading(false);
    }
  };

  // Transform data for react-force-graph-2d format
  const graphData = useMemo(() => {
    if (!result.graph_data || !result.graph_data.nodes.length) return { nodes: [], links: [] };

    const nodes = result.graph_data.nodes.map((node: any) => ({
      id: node.id,
      name: node.label,
      val: node.label === result.winning_diagnosis ? 35 : 18,
      color: getNodeColor(node.label, result.graph_data.links, result.winning_diagnosis),
    }));

    const links = result.graph_data.links
      .filter((link: any) => link.label !== 'BELONGS_TO')
      .map((link: any) => ({
        source: link.source,
        target: link.target,
        label: link.label.replace(/_/g, ' '),
        color: '#475569', // slightly lighter for better visibility
      }));

    return { nodes, links };
  }, [result.graph_data, result.winning_diagnosis]);

  const hasGraph = graphData.nodes.length > 0;
  const hasAbstracts = result.abstracts.length > 0;

  return (
    <div className="space-y-8 animate-fade-in">
      {/* ── Top Row: Response + Sidebar ── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* AI Response — 2/3 width */}
        <div className="lg:col-span-2">
          <div className="glass-card-strong p-6 sm:p-8">
            {/* Header */}
            <div className="flex items-center gap-3 mb-6">
              <div className="w-1 h-6 rounded-full bg-gradient-to-b from-curo-accent to-curo-teal shrink-0" />
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-curo-accent to-curo-teal flex items-center justify-center">
                <Stethoscope size={20} className="text-white" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-curo-text">Clinical Analysis</h2>
                <p className="text-sm text-curo-text-dim">Powered by CURO RAG Pipeline</p>
              </div>
            </div>

            {/* Response body */}
            <div
              className="accent-border-l cursor-pointer"
              onClick={!done ? skipToEnd : undefined}
              title={!done ? 'Click to skip animation' : undefined}
            >
              <div className="prose-sm">
                {done ? parseResponse(displayed) : (
                  <p className="leading-relaxed text-curo-text/90 text-lg">
                    {displayed}
                    <span className="typewriter-cursor" />
                  </p>
                )}
              </div>
            </div>

            {!done && (
              <p className="text-xs text-curo-text-dim mt-4 text-right italic">
                Click text to skip animation
              </p>
            )}

            {!hideChat && (
              <>
                {/* Follow-up Conversation */}
                {messages.length > 0 && (
                  <div className="mt-8 space-y-6 pt-6 border-t border-curo-border">
                    {messages.map((m, idx) => (
                      <div key={idx} className={`flex gap-4 ${m.role === 'user' ? 'justify-end' : ''}`}>
                        {m.role === 'assistant' && (
                          <div className="w-8 h-8 rounded-lg bg-curo-purple/20 flex items-center justify-center shrink-0">
                            <Stethoscope size={14} className="text-curo-purple" />
                          </div>
                        )}
                        <div className={`max-w-[85%] p-4 rounded-2xl text-sm leading-relaxed ${
                          m.role === 'user' 
                            ? 'bg-curo-accent/10 border border-curo-accent/20 text-curo-text' 
                            : 'bg-white/[0.03] border border-curo-border text-curo-text/90'
                        }`}>
                          {m.content}
                        </div>
                      </div>
                    ))}
                    {chatLoading && (
                      <div className="flex gap-4 animate-pulse">
                        <div className="w-8 h-8 rounded-lg bg-curo-purple/20 flex items-center justify-center shrink-0">
                          <div className="w-4 h-4 rounded-full bg-curo-purple/40 animate-bounce" />
                        </div>
                        <div className="h-12 w-2/3 bg-white/[0.02] rounded-2xl border border-curo-border" />
                      </div>
                    )}
                    <div ref={chatEndRef} />
                  </div>
                )}

                {/* Chat Input */}
                <form onSubmit={handleSendMessage} className="mt-8 flex gap-3">
                  <input
                    type="text"
                    placeholder="Ask a clinical follow-up question..."
                    className="flex-1 curo-input text-sm px-4 h-11"
                    value={newMessage}
                    onChange={(e) => setNewMessage(e.target.value)}
                    disabled={chatLoading}
                  />
                  <button 
                    type="submit" 
                    disabled={chatLoading || !newMessage.trim()}
                    className="w-11 h-11 flex items-center justify-center rounded-xl bg-curo-accent/10 border border-curo-accent/30 text-curo-accent hover:bg-curo-accent/20 transition-all disabled:opacity-50"
                  >
                    {chatLoading ? <RefreshCw size={18} className="animate-spin" /> : <MessageSquare size={18} />}
                  </button>
                </form>
              </>
            )}
          </div>
        </div>

        {/* Sidebar — 1/3 width */}
        <div className="space-y-6">
          {/* Winning Diagnosis */}
          <div className="glass-card p-5 relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-curo-accent/5 to-curo-purple/5 pointer-events-none" />
            <div className="relative">
              <div className="flex items-center gap-2 mb-3">
                <Award size={16} className="text-curo-accent" />
                <h3 className="text-sm font-semibold text-curo-text-muted uppercase tracking-wider">
                  Primary Diagnosis
                </h3>
              </div>
              <p className="text-xl font-bold gradient-text leading-snug">
                {result.winning_diagnosis}
              </p>
            </div>
          </div>

          {/* Differential Diagnosis */}
          <div className="glass-card p-5">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Shield size={16} className="text-curo-purple" />
                <h3 className="text-sm font-semibold text-curo-text-muted uppercase tracking-wider">
                  Differential Diagnosis
                </h3>
              </div>
              <button 
                type="button"
                onClick={onBack}
                className="text-[10px] uppercase tracking-wider font-bold text-curo-accent hover:text-curo-accent/80 transition-colors"
              >
                New Analysis
              </button>
            </div>
            <div className="space-y-2">
              {result.extracted_ddx.map((dx, i) => {
                const isWinner = dx === result.winning_diagnosis;
                return (
                  <a
                    key={i}
                    href={`https://www.google.com/search?q=${encodeURIComponent(dx)}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={`group flex items-center gap-3 p-3 rounded-lg transition-all animate-slide-in-right cursor-pointer hover:shadow-lg hover:scale-[1.02] ${
                      isWinner
                        ? 'bg-curo-accent/10 border border-curo-accent/30 hover:border-curo-accent'
                        : 'bg-white/[0.03] border border-curo-border hover:border-curo-teal/50 hover:bg-[#0f1729]'
                    }`}
                    style={{ animationDelay: `${i * 0.15}s`, opacity: 0 }}
                  >
                    <span
                      className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0 transition-colors ${
                        isWinner
                          ? 'bg-curo-accent/20 text-curo-accent'
                          : 'bg-curo-border text-curo-text-dim group-hover:bg-curo-teal/20 group-hover:text-curo-teal'
                      }`}
                    >
                      {i + 1}
                    </span>
                    <span className={`text-base font-medium transition-colors ${isWinner ? 'text-curo-accent' : 'text-curo-text-muted group-hover:text-curo-text'}`}>
                      {dx}
                    </span>
                    <Search size={14} className={`ml-auto opacity-0 group-hover:opacity-100 transition-opacity ${isWinner ? 'text-curo-accent' : 'text-curo-teal'}`} />
                  </a>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      {/* ── Knowledge Graph ── */}
      {hasGraph && (
        <div className="glass-card-strong p-6 sm:p-8 animate-slide-up" style={{ animationDelay: '0.3s', opacity: 0 }}>
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 rounded-xl bg-curo-purple/15 flex items-center justify-center">
              <Network size={20} className="text-curo-purple" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-curo-text">Knowledge Graph</h2>
              <p className="text-sm text-curo-text-dim">Clinical entity relationships extracted from medical literature</p>
            </div>
          </div>

          {/* Legend */}
          <div className="flex flex-wrap gap-4 mb-4 text-xs">
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-full bg-curo-purple inline-block" />
              <span className="text-curo-text-muted">Primary Diagnosis</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-full bg-curo-accent inline-block" />
              <span className="text-curo-text-muted">Disease / Entity</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-full bg-curo-amber inline-block" />
              <span className="text-curo-text-muted">Symptom / Trigger</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-3 rounded-full bg-curo-teal inline-block" />
              <span className="text-curo-text-muted">Treatment</span>
            </div>
          </div>

          <div className="h-[600px] sm:h-[750px] lg:h-[850px] w-full rounded-xl border border-curo-border bg-[#060a14] relative overflow-hidden" ref={containerRef}>
            {/* Graph stats overlay */}
            <div className="absolute top-3 right-3 z-10 flex items-center gap-3">
              <div className="flex items-center gap-2 bg-[#060a14]/80 backdrop-blur-sm rounded-lg px-3 py-1.5 border border-curo-border text-[10px] text-curo-text-dim shadow-lg">
                <span>{graphData.nodes.length} nodes</span>
                <span>·</span>
                <span>{graphData.links.length} relations</span>
              </div>
              <button 
                onClick={handleDownloadGraph}
                className="flex items-center gap-2 bg-curo-teal/10 hover:bg-curo-teal/20 backdrop-blur-sm rounded-lg px-3 py-1.5 border border-curo-teal/30 text-[10px] text-curo-teal font-bold shadow-lg transition-all active:scale-95"
                title="Download Knowledge Graph as PNG"
              >
                <Download size={12} />
                <span>SAVE PNG</span>
              </button>
            </div>
            
            {dims.width > 0 && (
              <ForceGraph2D
                onRef={(inst: any) => { fgRef.current = inst; }}
                width={dims.width}
                height={dims.height}
                graphData={graphData}
                cooldownTicks={100}
                onEngineStop={() => fgRef.current?.zoomToFit(400, 50)}
                nodeLabel="name"
                nodeColor="color"
                nodeRelSize={4}
                linkColor="color"
                linkWidth={1.5}
                linkDirectionalArrowLength={4}
                linkDirectionalArrowRelPos={1}
                linkDirectionalParticles={2}
                linkDirectionalParticleWidth={1.5}
                linkDirectionalParticleColor={() => '#8b5cf6'}
                linkDirectionalParticleSpeed={0.01}
                linkTargetNodeMargin={(link: any) => {
                  const node = typeof link.target === 'object' ? link.target : null;
                  return node && node.val ? Math.sqrt(node.val) * 4 : 8;
                }}
                backgroundColor="#060a14"
                d3VelocityDecay={0.15}
                onNodeClick={(node: any) => {
                  if (onNodeClick) onNodeClick(node.id);
                }}
                nodePointerAreaPaint={(node: any, color: string, ctx: CanvasRenderingContext2D) => {
                   const radius = Math.sqrt(node.val) * 4;
                   // Add a generous hit area including the text space below it
                   ctx.fillStyle = color;
                   ctx.beginPath();
                   ctx.arc(node.x, node.y, radius + 15, 0, 2 * Math.PI, false);
                   ctx.fill();
                }}
                nodeCanvasObjectMode={() => 'after'}
                nodeCanvasObject={(node: any, ctx, globalScale) => {
                  const label = node.name;
                  const fontSize = 12 / globalScale;
                  ctx.font = `${fontSize}px Inter, sans-serif`;
                  ctx.textAlign = 'center';
                  ctx.textBaseline = 'middle';
                  ctx.fillStyle = '#f8fafc';
                  const radius = Math.sqrt(node.val) * 4;
                  ctx.fillText(label, node.x, node.y + radius + fontSize);
                }}
                linkCanvasObjectMode={() => 'after'}
                linkCanvasObject={(link: any, ctx, globalScale) => {
                  const MAX_FONT_SIZE = 12;
                  const fontSize = Math.min(MAX_FONT_SIZE, 8 / globalScale);
                  const start = link.source;
                  const end = link.target;
                  
                  // ignore unbound links
                  if (typeof start !== 'object' || typeof end !== 'object') return;
                  
                  // Label text
                  const label = link.label;
                  if (!label) return;

                  // calculate label positioning
                  const textPos = {
                    x: start.x + (end.x - start.x) / 2,
                    y: start.y + (end.y - start.y) / 2
                  };

                  const relLink = { x: end.x - start.x, y: end.y - start.y };
                  let textAngle = Math.atan2(relLink.y, relLink.x);
                  // maintain label vertical orientation
                  if (textAngle > Math.PI / 2) textAngle = -(Math.PI - textAngle);
                  if (textAngle < -Math.PI / 2) textAngle = -(-Math.PI - textAngle);

                  ctx.font = `${fontSize}px Inter, sans-serif`;
                  const bBox = ctx.measureText(label);
                  const textWidth = bBox.width;
                  
                  ctx.save();
                  ctx.translate(textPos.x, textPos.y);
                  ctx.rotate(textAngle);
                  
                  // Background to mask line
                  ctx.fillStyle = '#060a14'; // backgroundColor
                  ctx.fillRect(-textWidth / 2 - 2, -fontSize/2 - 1, textWidth + 4, fontSize + 2);
                  
                  ctx.textAlign = 'center';
                  ctx.textBaseline = 'middle';
                  ctx.fillStyle = '#94a3b8'; // Label color
                  ctx.fillText(label, 0, 0);
                  ctx.restore();
                }}
              />
            )}
          </div>
        </div>
      )}

      {/* ── Abstracts ── */}
      {hasAbstracts && (
        <div className="animate-slide-up" style={{ animationDelay: '0.5s', opacity: 0 }}>
          <div className="flex items-center gap-3 mb-5">
            <div className="w-10 h-10 rounded-xl bg-curo-teal/15 flex items-center justify-center">
              <BookOpen size={20} className="text-curo-teal" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Retrieved Literature</h2>
              <p className="text-sm text-curo-text-dim">
                {result.abstracts.length} relevant abstract{result.abstracts.length !== 1 ? 's' : ''} from OpenAlex
              </p>
            </div>
          </div>

          <div className="space-y-3">
            {result.abstracts.map((abstract, i) => (
              <AbstractAccordion
                key={i}
                title={abstract.title}
                abstract={abstract.abstract}
                url={abstract.url}
                index={i}
              />
            ))}
          </div>
        </div>
      )}

      {/* ── Disclaimer ── */}
      <div className="text-center py-6">
        <p className="text-xs text-curo-text-dim max-w-lg mx-auto leading-relaxed">
          ⚕ CURO AI provides information for educational purposes only and is not a substitute 
          for professional medical advice, diagnosis, or treatment. Always consult a qualified 
          healthcare provider.
        </p>
      </div>
    </div>
  );
}