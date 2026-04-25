'use client';

import { useState, useRef, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import {
  Activity,
  FileUp,
  FileText,
  Trash2,
  Search,
  Send,
  ArrowLeft,
  FolderOpen,
  Database,
  Sparkles,
  X,
  RefreshCw,
  Upload,
  CheckCircle2,
  AlertCircle,
  Brain,
  LogOut,
  Layers,
  MessageSquare,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { signOut } from 'firebase/auth';
import { auth } from '../lib/firebase';
import Navbar from './Navbar';

interface StoredDocument {
  filename: string;
  chunks: number;
}

interface RecordsListResponse {
  documents: StoredDocument[];
  total_chunks: number;
}

interface UploadResult {
  filename: string;
  status: 'success' | 'error';
  chunks?: number;
  characters?: number;
  message?: string;
}

interface SourceCitation {
  filename: string;
  chunk_text: string;
  chunk_index: number;
}

interface QueryResponse {
  answer: string;
  sources: SourceCitation[];
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceCitation[];
}

// --- Source Citation Card ---
function SourceCard({ source, index }: { source: SourceCitation; index: number }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="glass-card overflow-hidden animate-fade-in" style={{ animationDelay: `${index * 0.05}s` }}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-start gap-3 p-3 text-left hover:bg-white/[0.02] transition-colors"
      >
        <div className="w-6 h-6 rounded-md bg-curo-teal/15 flex items-center justify-center shrink-0 mt-0.5">
          <FileText size={12} className="text-curo-teal" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium text-curo-text truncate">{source.filename}</p>
          <p className="text-[10px] text-curo-text-dim mt-0.5">Chunk #{source.chunk_index + 1}</p>
        </div>
        {expanded ? (
          <ChevronUp size={14} className="text-curo-text-dim shrink-0 mt-1" />
        ) : (
          <ChevronDown size={14} className="text-curo-text-dim shrink-0 mt-1" />
        )}
      </button>
      {expanded && (
        <div className="px-3 pb-3 pt-0 animate-fade-in">
          <div className="pl-9">
            <p className="text-xs text-curo-text-muted leading-relaxed whitespace-pre-wrap">
              {source.chunk_text}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════
export default function HealthRecordsVault({ userId }: { userId: string }) {
  const router = useRouter();

  // Document Manager State
  const [documents, setDocuments] = useState<StoredDocument[]>([]);
  const [totalChunks, setTotalChunks] = useState(0);
  const [isLoadingDocs, setIsLoadingDocs] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<UploadResult[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [deletingFile, setDeletingFile] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Query Chat State
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [queryInput, setQueryInput] = useState('');
  const [isQuerying, setIsQuerying] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Fetch stored documents
  const fetchDocuments = useCallback(async () => {
    setIsLoadingDocs(true);
    try {
      const res = await fetch(`http://127.0.0.1:8000/api/records/list?user_id=${encodeURIComponent(userId)}`);
      if (res.ok) {
        const data: RecordsListResponse = await res.json();
        setDocuments(data.documents || []);
        setTotalChunks(data.total_chunks || 0);
      }
    } catch (err) {
      console.error('Failed to fetch documents:', err);
    } finally {
      setIsLoadingDocs(false);
    }
  }, [userId]);

  useEffect(() => {
    fetchDocuments();
  }, [fetchDocuments]);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isQuerying]);

  // Upload handler
  const handleUpload = async (files: FileList | File[]) => {
    const pdfFiles = Array.from(files).filter(f => f.name.toLowerCase().endsWith('.pdf'));
    if (pdfFiles.length === 0) {
      alert('Please select PDF files only.');
      return;
    }

    setIsUploading(true);
    setUploadProgress([]);

    const formData = new FormData();
    pdfFiles.forEach(file => formData.append('files', file));

    try {
      const res = await fetch(`http://127.0.0.1:8000/api/records/upload?user_id=${encodeURIComponent(userId)}`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail || `Upload failed (${res.status})`);
      }

      const data = await res.json();
      setUploadProgress(data.results || []);
      
      // Refresh document list
      await fetchDocuments();
    } catch (err: any) {
      console.error('Upload error:', err);
      setUploadProgress([{ filename: 'Upload', status: 'error', message: err.message }]);
    } finally {
      setIsUploading(false);
    }
  };

  // Delete handler
  const handleDelete = async (filename?: string) => {
    const confirmed = confirm(
      filename
        ? `Delete "${filename}" from your health records?`
        : 'Delete ALL health records? This cannot be undone.'
    );
    if (!confirmed) return;

    setDeletingFile(filename || '__all__');

    try {
      const res = await fetch('http://127.0.0.1:8000/api/records/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId, filename: filename || null }),
      });

      if (res.ok) {
        await fetchDocuments();
        if (!filename) {
          setMessages([]);
        }
      }
    } catch (err) {
      console.error('Delete error:', err);
    } finally {
      setDeletingFile(null);
    }
  };

  // Query handler
  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!queryInput.trim() || isQuerying) return;

    const userMsg = queryInput.trim();
    const updatedMessages: ChatMessage[] = [...messages, { role: 'user', content: userMsg }];
    setMessages(updatedMessages);
    setQueryInput('');
    setIsQuerying(true);

    try {
      const res = await fetch('http://127.0.0.1:8000/api/records/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMsg, user_id: userId }),
      });

      if (!res.ok) throw new Error('Query failed');

      const data: QueryResponse = await res.json();
      setMessages([
        ...updatedMessages,
        { role: 'assistant', content: data.answer, sources: data.sources },
      ]);
    } catch (err: any) {
      setMessages([
        ...updatedMessages,
        { role: 'assistant', content: `Query error: ${err.message || 'Connection failed.'}` },
      ]);
    } finally {
      setIsQuerying(false);
    }
  };

  // Drag & Drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };
  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files.length > 0) {
      handleUpload(e.dataTransfer.files);
    }
  };

  // Quick query suggestions
  const quickQueries = [
    'Summarize my latest blood work',
    'What medications am I on?',
    'Any abnormal test values?',
    'What diagnoses are documented?',
  ];

  return (
    <div className="min-h-screen flex flex-col">
      {/* ═══ Header ═══ */}
      <Navbar 
        extraContent={
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-curo-teal to-curo-accent flex items-center justify-center">
              <FolderOpen size={20} className="text-white" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                Health Records Vault
                <span className="text-[10px] text-curo-teal px-1.5 py-0.5 rounded bg-curo-teal/10 border border-curo-teal/20">
                  RAG
                </span>
              </h2>
              <div className="flex items-center gap-1.5 mt-0.5">
                <div className="w-1.5 h-1.5 rounded-full bg-curo-teal animate-pulse" />
                <span className="text-[10px] text-curo-text-dim uppercase tracking-wider">
                  {totalChunks > 0 ? `${documents.length} docs · ${totalChunks} chunks` : 'Records Vault Ready'}
                </span>
              </div>
            </div>
          </div>
        }
      />

      {/* ═══ Main Content ═══ */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 h-full">
          {/* ─── Left Panel: Documents ─── */}
          <div className="lg:col-span-2 space-y-5">
            {/* Upload Zone */}
            <div
              className={`glass-card-strong p-6 transition-all ${
                isDragging
                  ? 'border-curo-teal border-2 bg-curo-teal/5 shadow-lg shadow-curo-teal/10'
                  : ''
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-curo-teal to-curo-accent flex items-center justify-center">
                  <Upload size={20} className="text-white" />
                </div>
                <div>
                  <h2 className="text-sm font-bold text-curo-text">Upload Documents</h2>
                  <p className="text-[11px] text-curo-text-dim">PDF reports, lab results, prescriptions</p>
                </div>
              </div>

              {/* Drop Zone */}
              <div
                onClick={() => fileInputRef.current?.click()}
                className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all group ${
                  isDragging
                    ? 'border-curo-teal bg-curo-teal/5'
                    : 'border-curo-border hover:border-curo-teal/50 hover:bg-white/[0.02]'
                } ${isUploading ? 'opacity-50 pointer-events-none' : ''}`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".pdf"
                  className="hidden"
                  onChange={(e) => e.target.files && handleUpload(e.target.files)}
                  disabled={isUploading}
                />
                {isUploading ? (
                  <div className="flex flex-col items-center gap-3">
                    <RefreshCw size={28} className="text-curo-teal animate-spin" />
                    <p className="text-sm text-curo-teal font-medium">Processing PDFs…</p>
                    <p className="text-xs text-curo-text-dim">Extracting text, running OCR if needed</p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-3">
                    <div className="w-14 h-14 rounded-2xl bg-curo-teal/10 flex items-center justify-center group-hover:scale-110 transition-transform">
                      <FileUp size={24} className="text-curo-teal" />
                    </div>
                    <div>
                      <p className="text-sm text-curo-text font-medium">
                        Drop PDFs here or click to browse
                      </p>
                      <p className="text-xs text-curo-text-dim mt-1">
                        Supports text & scanned/image PDFs (OCR)
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Upload Results */}
              {uploadProgress.length > 0 && (
                <div className="mt-4 space-y-2 animate-fade-in">
                  {uploadProgress.map((result, i) => (
                    <div
                      key={i}
                      className={`flex items-center gap-3 p-3 rounded-lg text-sm ${
                        result.status === 'success'
                          ? 'bg-curo-teal/10 border border-curo-teal/20 text-curo-teal'
                          : 'bg-curo-rose/10 border border-curo-rose/20 text-curo-rose'
                      }`}
                    >
                      {result.status === 'success' ? (
                        <CheckCircle2 size={16} />
                      ) : (
                        <AlertCircle size={16} />
                      )}
                      <span className="flex-1 truncate text-xs font-medium">{result.filename}</span>
                      {result.status === 'success' && (
                        <span className="text-[10px] opacity-70">{result.chunks} chunks</span>
                      )}
                      {result.status === 'error' && (
                        <span className="text-[10px] opacity-70 truncate max-w-[120px]">{result.message}</span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Stored Documents List */}
            <div className="glass-card p-5">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Database size={16} className="text-curo-purple" />
                  <h3 className="text-xs font-semibold text-curo-text-muted uppercase tracking-wider">
                    Stored Records
                  </h3>
                </div>
                {documents.length > 0 && (
                  <button
                    onClick={() => handleDelete()}
                    disabled={deletingFile === '__all__'}
                    className="text-[10px] uppercase tracking-wider font-bold text-curo-rose/60 hover:text-curo-rose transition-colors disabled:opacity-50"
                  >
                    Clear All
                  </button>
                )}
              </div>

              {isLoadingDocs ? (
                <div className="flex items-center justify-center py-8">
                  <RefreshCw size={18} className="text-curo-text-dim animate-spin" />
                </div>
              ) : documents.length === 0 ? (
                <div className="text-center py-8">
                  <Layers size={28} className="text-curo-text-dim mx-auto mb-3 opacity-40" />
                  <p className="text-sm text-curo-text-dim">No documents uploaded yet</p>
                  <p className="text-xs text-curo-text-dim mt-1 opacity-60">
                    Upload health reports to get started
                  </p>
                </div>
              ) : (
                <div className="space-y-2">
                  {documents.map((doc, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-3 p-3 rounded-lg bg-white/[0.03] border border-curo-border group hover:border-curo-teal/30 transition-all animate-slide-in-right"
                      style={{ animationDelay: `${i * 0.08}s`, opacity: 0 }}
                    >
                      <div className="w-8 h-8 rounded-lg bg-curo-teal/10 flex items-center justify-center shrink-0">
                        <FileText size={14} className="text-curo-teal" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-curo-text truncate">{doc.filename}</p>
                        <p className="text-[10px] text-curo-text-dim">{doc.chunks} chunks indexed</p>
                      </div>
                      <button
                        onClick={() => handleDelete(doc.filename)}
                        disabled={deletingFile === doc.filename}
                        className="p-1.5 rounded-md opacity-0 group-hover:opacity-100 hover:bg-curo-rose/10 text-curo-text-dim hover:text-curo-rose transition-all disabled:opacity-50"
                      >
                        {deletingFile === doc.filename ? (
                          <RefreshCw size={14} className="animate-spin" />
                        ) : (
                          <Trash2 size={14} />
                        )}
                      </button>
                    </div>
                  ))}

                  {/* Stats */}
                  <div className="flex items-center justify-between pt-3 mt-2 border-t border-curo-border">
                    <span className="text-[10px] text-curo-text-dim uppercase tracking-wider font-bold">
                      Total
                    </span>
                    <div className="flex items-center gap-3 text-[10px] text-curo-text-dim">
                      <span>{documents.length} files</span>
                      <span>·</span>
                      <span>{totalChunks} chunks</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Integration Info */}
            <div className="glass-card p-4">
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-lg bg-curo-accent/10 flex items-center justify-center shrink-0">
                  <Sparkles size={14} className="text-curo-accent" />
                </div>
                <div>
                  <p className="text-xs font-semibold text-curo-text mb-1">Auto-Integrated</p>
                  <p className="text-[11px] text-curo-text-dim leading-relaxed">
                    Your uploaded records are automatically used when you run symptom analysis on the Dashboard.
                    CURO AI retrieves relevant sections from your records to enhance clinical insights.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* ─── Right Panel: Query Chat ─── */}
          <div className="lg:col-span-3 flex flex-col">
            <div className="glass-card-strong flex flex-col flex-1 overflow-hidden" style={{ minHeight: '600px' }}>
              {/* Chat Header */}
              <div className="flex items-center gap-3 p-5 border-b border-curo-border">
                <div className="w-10 h-10 rounded-xl bg-curo-purple/15 flex items-center justify-center">
                  <Search size={20} className="text-curo-purple" />
                </div>
                <div>
                  <h2 className="text-sm font-bold text-curo-text">Query Your Records</h2>
                  <p className="text-[11px] text-curo-text-dim">
                    Ask questions about your uploaded health documents
                  </p>
                </div>
              </div>

              {/* Chat Messages */}
              <div className="flex-1 overflow-y-auto p-5 space-y-5">
                {messages.length === 0 && !isQuerying && (
                  <div className="flex flex-col items-center justify-center h-full text-center py-12 animate-fade-in">
                    <div className="w-16 h-16 rounded-2xl bg-curo-purple/10 flex items-center justify-center mb-4">
                      <MessageSquare size={28} className="text-curo-purple/40" />
                    </div>
                    <h3 className="text-base font-semibold text-curo-text mb-2">
                      Ask About Your Records
                    </h3>
                    <p className="text-xs text-curo-text-dim max-w-sm leading-relaxed mb-6">
                      Upload your health reports above, then ask questions about your test results,
                      medications, diagnoses, or any medical information in your documents.
                    </p>

                    {/* Quick Query Suggestions */}
                    {documents.length > 0 && (
                      <div className="flex flex-wrap gap-2 justify-center max-w-md">
                        {quickQueries.map((q, i) => (
                          <button
                            key={i}
                            onClick={() => {
                              setQueryInput(q);
                            }}
                            className="px-3 py-1.5 rounded-full border border-curo-purple/20 bg-curo-purple/5 text-xs text-curo-purple hover:bg-curo-purple hover:text-white transition-all hover:scale-105 active:scale-95"
                          >
                            {q}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {messages.map((msg, i) => (
                  <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}>
                    {msg.role === 'assistant' && (
                      <div className="w-8 h-8 rounded-full bg-curo-purple/20 flex items-center justify-center mr-3 shrink-0 mt-1">
                        <Search size={14} className="text-curo-purple" />
                      </div>
                    )}
                    <div className={`max-w-[85%] ${msg.role === 'user' ? '' : ''}`}>
                      <div
                        className={`p-4 rounded-2xl text-sm leading-relaxed ${
                          msg.role === 'user'
                            ? 'bg-curo-purple/10 border border-curo-purple/20 text-curo-text rounded-tr-sm'
                            : 'bg-white/[0.03] border border-curo-border text-curo-text/90 rounded-tl-sm'
                        }`}
                      >
                        <div className="whitespace-pre-wrap">{msg.content}</div>
                      </div>

                      {/* Source Citations */}
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="mt-2 space-y-1">
                          <p className="text-[10px] text-curo-text-dim uppercase tracking-wider font-bold ml-1 mb-1">
                            Sources ({msg.sources.length})
                          </p>
                          {msg.sources.map((source, j) => (
                            <SourceCard key={j} source={source} index={j} />
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {isQuerying && (
                  <div className="flex justify-start animate-fade-in">
                    <div className="w-8 h-8 rounded-full bg-curo-purple/20 flex items-center justify-center mr-3 shrink-0">
                      <Search size={14} className="text-curo-purple" />
                    </div>
                    <div className="bg-white/[0.05] border border-curo-border p-4 rounded-2xl rounded-tl-sm">
                      <div className="flex gap-1.5">
                        <div className="w-2 h-2 bg-curo-purple/60 rounded-full animate-bounce [animation-delay:-0.3s]" />
                        <div className="w-2 h-2 bg-curo-purple/60 rounded-full animate-bounce [animation-delay:-0.15s]" />
                        <div className="w-2 h-2 bg-curo-purple/60 rounded-full animate-bounce" />
                      </div>
                    </div>
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>

              {/* Chat Input */}
              <div className="border-t border-curo-border p-4">
                <form onSubmit={handleQuery} className="flex gap-3">
                  <input
                    type="text"
                    placeholder={
                      documents.length === 0
                        ? 'Upload documents first to start querying…'
                        : 'Ask about your health records…'
                    }
                    className="flex-1 curo-input text-sm px-4 h-11"
                    value={queryInput}
                    onChange={(e) => setQueryInput(e.target.value)}
                    disabled={isQuerying || documents.length === 0}
                  />
                  <button
                    type="submit"
                    disabled={isQuerying || !queryInput.trim() || documents.length === 0}
                    className="w-11 h-11 flex items-center justify-center rounded-xl bg-curo-purple/10 border border-curo-purple/30 text-curo-purple hover:bg-curo-purple/20 transition-all disabled:opacity-50"
                  >
                    {isQuerying ? (
                      <RefreshCw size={18} className="animate-spin" />
                    ) : (
                      <Send size={18} />
                    )}
                  </button>
                </form>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* ═══ Footer ═══ */}
      <footer className="py-4 text-center text-xs text-curo-text-dim">
        <p>⚕ Your health records are stored locally and never shared. CURO AI is not a substitute for professional medical advice.</p>
      </footer>
    </div>
  );
}
