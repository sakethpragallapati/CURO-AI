'use client';

import { useState, useEffect } from 'react';
import { onAuthStateChanged, User, signOut } from 'firebase/auth';
import { auth } from '../lib/firebase';
import { useRouter } from 'next/navigation';
import { Activity, Sparkles, Brain, Search, Shield, FlaskConical, Stethoscope, ChevronRight, Zap, BarChart3, Network, LogOut, FolderOpen } from 'lucide-react';
import AuthModal from '../components/AuthModal';
import StarBorder from '../components/StarBorder';
import LogoLoop from '../components/LogoLoop';

const features = [
  { icon: Search, title: 'Evidence Retrieval', desc: 'Fetches and ranks peer-reviewed abstracts from OpenAlex in real-time, filtered by clinical authority.', color: 'curo-accent' },
  { icon: Network, title: 'Knowledge Graph', desc: 'Automatically extracts clinical entity relationships and visualizes them as an interactive 3D graph.', color: 'curo-purple' },
  { icon: Brain, title: 'Agentic Triage', desc: 'AI-driven clinical interview that dynamically adapts questions based on your responses.', color: 'curo-teal' },
  { icon: Stethoscope, title: 'Differential Diagnosis', desc: 'Generates and ranks differential diagnoses using an LLM triage router with acuity prioritization.', color: 'curo-accent' },
  { icon: Shield, title: 'Semantic Firewall', desc: 'Multi-query retrieval with contextual compression ensures only relevant evidence reaches synthesis.', color: 'curo-purple' },
  { icon: BarChart3, title: 'Clinical Guidelines', desc: 'Synthesizes treatment protocols from NICE, CDC, and NIH guidelines for evidence-based pathways.', color: 'curo-teal' },
  { icon: FolderOpen, title: 'Health Records Vault', desc: 'Upload multiple PDFs (including scanned reports via OCR) and query your medical history with RAG.', color: 'curo-accent' },
];

const featureLogos = features.map((feature, i) => ({
  node: <span key={i} />,
  title: feature.title,
}));

function FeatureCard({ feature }: { feature: typeof features[number] }) {
  const Icon = feature.icon;
  return (
    <StarBorder color="magenta" speed="5s" thickness={2}>
      <div className="p-6 group" style={{ minWidth: '280px', maxWidth: '320px' }}>
        <div className={`w-11 h-11 rounded-xl bg-${feature.color}/15 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
          <Icon size={22} className={`text-${feature.color}`} />
        </div>
        <h3 className="text-sm font-bold text-curo-text mb-2">{feature.title}</h3>
        <p className="text-xs text-curo-text-muted leading-relaxed">{feature.desc}</p>
      </div>
    </StarBorder>
  );
}

export default function LandingPage() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [showAuth, setShowAuth] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setLoading(false);
    });
    return () => unsubscribe();
  }, []);

  const handleGetStarted = () => {
    if (user) {
      router.push('/chat');
    } else {
      setShowAuth(true);
    }
  };

  const handleAuthSuccess = () => {
    setShowAuth(false);
    router.push('/chat');
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="flex items-center gap-3 text-curo-text-dim">
          <div className="w-5 h-5 rounded-full bg-curo-accent animate-pulse" />
          <span className="text-sm">Initializing CURO AI...</span>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="min-h-screen flex flex-col relative overflow-hidden">
        {/* Nav */}
        <nav className="relative z-20 py-5 px-6 sm:px-10">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-curo-accent to-curo-teal flex items-center justify-center shadow-lg shadow-curo-accent/20">
                <Activity size={20} className="text-white" />
              </div>
              <span className="text-xl font-bold text-white tracking-tight">CURO <span className="gradient-text">AI</span></span>
            </div>
            <div className="flex items-center gap-4">
              {user ? (
                <>
                  <button onClick={() => router.push('/records')} className="text-sm text-curo-text-muted hover:text-curo-teal transition-colors hidden sm:block">
                    Health Records
                  </button>
                  <button onClick={() => router.push('/triage')} className="text-sm text-curo-text-muted hover:text-curo-accent transition-colors hidden sm:block">
                    Triage Assistant
                  </button>
                  <button onClick={() => router.push('/chat')} className="glow-btn text-sm py-2.5 px-5">
                    Open Dashboard
                  </button>
                  <button onClick={async () => { await signOut(auth); setShowAuth(false); }} className="p-2.5 rounded-lg border border-curo-border hover:border-curo-rose/50 text-curo-text-dim hover:text-curo-rose transition-all flex items-center justify-center gap-2" title="Sign Out">
                    <LogOut size={16} />
                  </button>
                </>
              ) : (
                <button onClick={() => setShowAuth(true)} className="text-sm text-curo-text-muted hover:text-white border border-curo-border rounded-lg px-4 py-2 transition-all hover:border-curo-accent/50">
                  Sign In
                </button>
              )}
            </div>
          </div>
        </nav>

        {/* Hero Section */}
        <section className="relative z-10 flex-1 flex items-center justify-center px-6 py-16 sm:py-24">
          <div className="max-w-5xl mx-auto text-center space-y-8 animate-fade-in">
            {/* Badge */}
            <div className="inline-flex items-center gap-2 curo-pill curo-pill-active text-xs sm:text-sm animate-slide-up" style={{ animationDelay: '0.1s', opacity: 0 }}>
              <Zap size={14} />
              Agentic Clinical Decision Support — Powered by RAG
            </div>

            {/* Headline */}
            <h1 className="text-4xl sm:text-6xl lg:text-7xl font-bold text-white leading-[1.1] tracking-tight animate-slide-up" style={{ animationDelay: '0.2s', opacity: 0 }}>
              Intelligent{' '}
              <span className="gradient-text">Symptom Analysis</span>
              <br className="hidden sm:block" />
              {' '}Backed by Evidence
            </h1>

            {/* Subtitle */}
            <p className="text-base sm:text-lg text-curo-text-muted max-w-2xl mx-auto leading-relaxed animate-slide-up" style={{ animationDelay: '0.3s', opacity: 0 }}>
              CURO AI retrieves real-time medical literature, constructs clinical knowledge graphs, 
              and synthesizes evidence-based insights — all through an intelligent conversational interface.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-slide-up" style={{ animationDelay: '0.4s', opacity: 0 }}>
              <button onClick={handleGetStarted} className="glow-btn text-base py-4 px-8 flex items-center gap-3 group">
                <Sparkles size={20} />
                <span>Get Started</span>
                <ChevronRight size={18} className="transition-transform group-hover:translate-x-1" />
              </button>
              <button onClick={() => user ? router.push('/triage') : setShowAuth(true)} className="flex items-center gap-3 px-8 py-4 rounded-xl border border-curo-border bg-white/[0.02] text-curo-text hover:bg-white/[0.05] hover:border-curo-accent/30 transition-all text-base">
                <Brain size={20} className="text-curo-accent" />
                <span>Try Curo Assistant</span>
              </button>
            </div>
          </div>
        </section>

        {/* Features — Scrolling LogoLoop Carousel */}
        <section className="relative z-10 pb-20 animate-slide-up" style={{ animationDelay: '0.6s', opacity: 0 }}>
          <div className="max-w-7xl mx-auto px-6 mb-8">
            <h2 className="text-lg sm:text-xl font-semibold text-white text-center mb-2">Powered by Cutting-Edge AI</h2>
            <p className="text-sm text-curo-text-muted text-center">Explore our core capabilities</p>
          </div>
          <div style={{ height: '220px', position: 'relative', overflow: 'hidden' }}>
            <LogoLoop
              logos={featureLogos}
              speed={60}
              direction="left"
              logoHeight={200}
              gap={24}
              hoverSpeed={0}
              fadeOut
              fadeOutColor="#0a0e1a"
              ariaLabel="CURO AI Features"
              renderItem={(item, key) => {
                const idx = parseInt(String(key).split('-')[1]);
                const feature = features[idx % features.length];
                return <FeatureCard feature={feature} />;
              }}
            />
          </div>
        </section>

        {/* Footer */}
        <footer className="relative z-10 py-8 text-center border-t border-curo-border/50">
          <p className="text-xs text-curo-text-dim">
            ⚕ CURO AI is not a substitute for professional medical advice, diagnosis, or treatment.
          </p>
        </footer>
      </div>

      {/* Auth Modal */}
      {showAuth && <AuthModal onClose={() => setShowAuth(false)} onSuccess={handleAuthSuccess} />}
    </>
  );
}