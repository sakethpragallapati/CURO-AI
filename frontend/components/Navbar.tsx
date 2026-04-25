'use client';

import { useState, useEffect } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { Activity, FolderOpen, Brain, LogOut, History, User, ChevronDown } from 'lucide-react';
import { auth } from '../lib/firebase';
import { onAuthStateChanged, User as FirebaseUser, signOut } from 'firebase/auth';

interface NavbarProps {
  showLinks?: boolean;
  extraContent?: React.ReactNode;
  onSignInClick?: () => void;
}

export default function Navbar({ showLinks = true, extraContent, onSignInClick }: NavbarProps) {
  const router = useRouter();
  const pathname = usePathname();
  const [user, setUser] = useState<FirebaseUser | null>(null);
  const [isProfileOpen, setIsProfileOpen] = useState(false);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
    });
    return () => unsubscribe();
  }, []);

  const navLinks = [
    { name: 'Dashboard', path: '/chat', icon: Activity },
    { name: 'Health Vault', path: '/records', icon: FolderOpen },
    { name: 'Curo Assistant', path: '/triage', icon: Brain },
  ];

  return (
    <nav className="sticky top-0 z-50 w-full glass-card-strong border-t-0 border-x-0 rounded-none h-20 flex items-center px-6 sm:px-10">
      <div className="w-full max-w-[95%] 2xl:max-w-[1600px] mx-auto flex items-center justify-between">
        {/* Left: Logo & Extra Content */}
        <div className="flex items-center gap-8">
          <div 
            className="flex items-center gap-3 cursor-pointer group transition-all" 
            onClick={() => router.push('/')}
          >
            <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-curo-accent to-curo-teal flex items-center justify-center shadow-lg shadow-curo-accent/20 group-hover:scale-105 transition-transform">
              <Activity size={22} className="text-white" />
            </div>
            <span className="text-3xl font-bold text-white tracking-tight">
              CURO <span className="gradient-text">AI</span>
            </span>
          </div>

          {extraContent && (
            <div className="hidden lg:flex items-center border-l border-curo-border pl-8 ml-2">
              {extraContent}
            </div>
          )}
        </div>

        {/* Center/Right: Navigation Links */}
        <div className="flex items-center gap-4 sm:gap-6">
          {showLinks && user && (
            <div className="hidden md:flex items-center gap-1">
              {navLinks.map((link) => {
                const Icon = link.icon;
                const isActive = pathname === link.path;
                return (
                  <button
                    key={link.path}
                    onClick={() => router.push(link.path)}
                    className={`flex items-center gap-2.5 px-5 py-2.5 rounded-xl transition-all text-lg font-medium ${
                      isActive
                        ? 'bg-white/[0.08] text-white'
                        : 'text-curo-text-dim hover:text-white hover:bg-white/[0.04]'
                    }`}
                  >
                    <Icon size={20} className={isActive ? 'text-curo-accent' : ''} />
                    {link.name}
                  </button>
                );
              })}
            </div>
          )}

          {/* User Profile / Auth */}
          <div className="flex items-center gap-3 pl-2 border-l border-curo-border/50">
            {user ? (
              <div className="relative">
                <button 
                  onClick={() => setIsProfileOpen(!isProfileOpen)}
                  className="flex items-center gap-2 p-1 rounded-full border border-curo-border bg-white/[0.02] hover:bg-white/[0.05] transition-all"
                >
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-curo-accent/20 to-curo-teal/20 flex items-center justify-center text-curo-accent">
                    <User size={22} />
                  </div>
                  <ChevronDown size={14} className={`text-curo-text-dim mr-1 transition-transform ${isProfileOpen ? 'rotate-180' : ''}`} />
                </button>

                {isProfileOpen && (
                  <div className="absolute right-0 mt-3 w-56 glass-card-strong p-2 shadow-2xl animate-fade-in border border-curo-border">
                    <div className="px-4 py-3 border-b border-curo-border mb-1">
                      <p className="text-sm font-bold text-white truncate">{user.displayName || 'User'}</p>
                      <p className="text-xs text-curo-text-dim truncate">{user.email}</p>
                    </div>
                    <button 
                      onClick={async () => { await signOut(auth); setIsProfileOpen(false); router.push('/'); }}
                      className="w-full flex items-center gap-3 px-4 py-2.5 text-sm font-medium text-curo-rose hover:bg-curo-rose/10 rounded-lg transition-colors"
                    >
                      <LogOut size={16} />
                      Sign Out
                    </button>
                  </div>
                )}
              </div>
            ) : (
              <button 
                onClick={() => {
                  if (onSignInClick) {
                    onSignInClick();
                  } else {
                    router.push('/');
                  }
                }}
                className="text-lg text-curo-text-muted hover:text-white border border-curo-border rounded-xl px-6 py-2.5 transition-all hover:border-curo-accent/50"
              >
                Sign In
              </button>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
