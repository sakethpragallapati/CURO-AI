'use client';

import { useRef, useEffect, useCallback } from 'react';
import gsap from 'gsap';
import './DotGrid.css';

interface DotGridProps {
  dotSize?: number;
  gap?: number;
  baseColor?: string;
  activeColor?: string;
  proximity?: number;
  shockRadius?: number;
  shockStrength?: number;
  resistance?: number;
  returnDuration?: number;
}

interface Dot {
  baseX: number;
  baseY: number;
  x: number;
  y: number;
  dx: number;
  dy: number;
}

const DotGrid = ({
  dotSize = 5,
  gap = 15,
  baseColor = '#2F293A',
  activeColor = '#5227FF',
  proximity = 120,
  shockRadius = 250,
  shockStrength = 5,
  resistance = 750,
  returnDuration = 1.5,
}: DotGridProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const dotsRef = useRef<Dot[]>([]);
  const mouseRef = useRef({ x: -1000, y: -1000 });
  const rafRef = useRef<number | null>(null);

  const initDots = useCallback(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const { clientWidth: w, clientHeight: h } = wrap;
    const cols = Math.floor(w / (dotSize + gap));
    const rows = Math.floor(h / (dotSize + gap));
    const offsetX = (w - cols * (dotSize + gap) + gap) / 2;
    const offsetY = (h - rows * (dotSize + gap) + gap) / 2;
    const dots: Dot[] = [];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = offsetX + c * (dotSize + gap);
        const y = offsetY + r * (dotSize + gap);
        dots.push({ baseX: x, baseY: y, x, y, dx: 0, dy: 0 });
      }
    }
    dotsRef.current = dots;
  }, [dotSize, gap]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const mx = mouseRef.current.x;
    const my = mouseRef.current.y;

    for (const dot of dotsRef.current) {
      const drawX = dot.baseX + dot.dx;
      const drawY = dot.baseY + dot.dy;
      const dist = Math.sqrt((mx - drawX) ** 2 + (my - drawY) ** 2);
      const t = Math.max(0, 1 - dist / proximity);
      ctx.beginPath();
      ctx.arc(drawX, drawY, dotSize / 2, 0, Math.PI * 2);
      ctx.fillStyle = t > 0 ? activeColor : baseColor;
      ctx.globalAlpha = t > 0 ? 0.3 + t * 0.7 : 0.3;
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    rafRef.current = requestAnimationFrame(draw);
  }, [dotSize, baseColor, activeColor, proximity]);

  const handleResize = useCallback(() => {
    const wrap = wrapRef.current;
    const canvas = canvasRef.current;
    if (!wrap || !canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = wrap.clientWidth * dpr;
    canvas.height = wrap.clientHeight * dpr;
    canvas.style.width = `${wrap.clientWidth}px`;
    canvas.style.height = `${wrap.clientHeight}px`;
    const ctx = canvas.getContext('2d');
    if (ctx) ctx.scale(dpr, dpr);
    initDots();
  }, [initDots]);

  // Use window-level mouse tracking so events work even when content is layered on top
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      mouseRef.current = {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      };
    };

    const handleClick = (e: MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;

      for (const dot of dotsRef.current) {
        const dist = Math.sqrt((cx - dot.baseX) ** 2 + (cy - dot.baseY) ** 2);
        if (dist < shockRadius) {
          const angle = Math.atan2(dot.baseY - cy, dot.baseX - cx);
          const force = (1 - dist / shockRadius) * shockStrength * (dotSize + gap);
          gsap.killTweensOf(dot);
          dot.dx = Math.cos(angle) * force;
          dot.dy = Math.sin(angle) * force;
          gsap.to(dot, {
            dx: 0,
            dy: 0,
            duration: returnDuration,
            ease: 'elastic.out(1, 0.3)',
            delay: (dist / shockRadius) * 0.2,
          });
        }
      }
    };

    const handleMouseLeave = () => {
      mouseRef.current = { x: -1000, y: -1000 };
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('click', handleClick);
    document.addEventListener('mouseleave', handleMouseLeave);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('click', handleClick);
      document.removeEventListener('mouseleave', handleMouseLeave);
    };
  }, [shockRadius, shockStrength, dotSize, gap, returnDuration]);

  useEffect(() => {
    handleResize();
    rafRef.current = requestAnimationFrame(draw);
    window.addEventListener('resize', handleResize);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      window.removeEventListener('resize', handleResize);
    };
  }, [handleResize, draw]);

  return (
    <div className="dot-grid">
      <div ref={wrapRef} className="dot-grid__wrap">
        <canvas ref={canvasRef} className="dot-grid__canvas" />
      </div>
    </div>
  );
};

export default DotGrid;
