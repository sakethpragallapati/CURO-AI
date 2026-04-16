import type { Metadata, Viewport } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import DotGridBackground from '../components/DotGridBackground'

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
})

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
}

export const metadata: Metadata = {
  title: 'CURO AI — Clinical RAG Assistant',
  description: 'An AI-powered clinical decision support system that uses Retrieval-Augmented Generation to analyze symptoms, retrieve medical literature, and provide evidence-based differential diagnoses.',
  keywords: ['clinical AI', 'medical diagnosis', 'RAG', 'differential diagnosis', 'symptom analysis'],
  authors: [{ name: 'CURO AI' }],
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="font-sans min-h-screen bg-curo-bg text-curo-text antialiased">
        {/* Animated mesh blobs (kept for ambient color) */}
        <div className="curo-bg-mesh" aria-hidden="true" />
        {/* Interactive dot grid background */}
        <DotGridBackground />
        
        {/* Fixed blur overlay to soften the dots without creating a containing block for content */}
        <div className="fixed inset-0 z-0 pointer-events-none" style={{ backdropFilter: 'blur(0.5px)' }} aria-hidden="true" />

        {/* Main content */}
        <div className="relative z-10">
          {children}
        </div>
      </body>
    </html>
  )
}