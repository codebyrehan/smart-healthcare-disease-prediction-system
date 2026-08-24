/**
 * Interactive 3D Medical AI Core & Neural Visualization Canvas
 * High-performance lightweight 3D canvas with orbital particle nodes & mouse parallax tilt.
 */

(function () {
  const canvas = document.getElementById('medical-ai-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  // Device pixel ratio for crisp high-DPI rendering
  const dpr = window.devicePixelRatio || 1;
  const size = 500;
  canvas.width = size * dpr;
  canvas.height = size * dpr;
  ctx.scale(dpr, dpr);

  const orbHost = document.querySelector('.health-orb');
  const heroSection = document.querySelector('.hero-3d');

  // Sphere and Node parameters
  const center = { x: 250, y: 250 };
  const radius = 130;
  const nodeCount = 42;
  const nodes = [];

  // Generate 3D Fibonacci sphere distribution for uniform spherical coverage
  const phi = Math.PI * (3 - Math.sqrt(5)); // Golden ratio angle
  for (let i = 0; i < nodeCount; i++) {
    const y = 1 - (i / (nodeCount - 1)) * 2; // y goes from 1 to -1
    const r = Math.sqrt(1 - y * y); // radius at y
    const theta = phi * i;

    const x = Math.cos(theta) * r;
    const z = Math.sin(theta) * r;

    nodes.push({
      x: x * radius,
      y: y * radius,
      z: z * radius,
      baseX: x * radius,
      baseY: y * radius,
      baseZ: z * radius,
      size: Math.random() * 2.5 + 2,
      pulseSpeed: Math.random() * 0.04 + 0.02,
      pulsePhase: Math.random() * Math.PI * 2,
    });
  }

  // Rotation angles and velocities
  let rotX = 0.2;
  let rotY = 0.4;
  let targetRotX = 0.2;
  let targetRotY = 0.4;
  let pulseTime = 0;
  let isHovered = false;

  // Mouse Parallax tracking
  if (heroSection) {
    heroSection.addEventListener('pointermove', (e) => {
      const rect = heroSection.getBoundingClientRect();
      const nx = (e.clientX - rect.left) / rect.width - 0.5;
      const ny = (e.clientY - rect.top) / rect.height - 0.5;

      targetRotY = nx * 1.4;
      targetRotX = -ny * 1.1;
      isHovered = true;

      if (orbHost) {
        orbHost.style.transform = `rotateX(${ny * -15}deg) rotateY(${nx * 20}deg)`;
      }
    });

    heroSection.addEventListener('pointerleave', () => {
      targetRotX = 0.2;
      targetRotY = 0.4;
      isHovered = false;
      if (orbHost) {
        orbHost.style.transform = 'rotateX(0deg) rotateY(0deg)';
      }
    });
  }

  // Animation Loop
  function render() {
    ctx.clearRect(0, 0, size, size);

    // Smooth rotation interpolation
    rotX += (targetRotX - rotX) * 0.05 + 0.002;
    rotY += (targetRotY - rotY) * 0.05 + 0.005;
    pulseTime += 0.03;

    // Draw central glowing biometric core
    const coreGlow = ctx.createRadialGradient(center.x, center.y, 10, center.x, center.y, 140);
    coreGlow.addColorStop(0, 'rgba(6, 182, 212, 0.25)');
    coreGlow.addColorStop(0.5, 'rgba(16, 185, 129, 0.12)');
    coreGlow.addColorStop(1, 'rgba(6, 182, 212, 0)');
    ctx.fillStyle = coreGlow;
    ctx.beginPath();
    ctx.arc(center.x, center.y, 140, 0, Math.PI * 2);
    ctx.fill();

    // Pulse core ring
    const corePulseR = 55 + Math.sin(pulseTime * 1.5) * 8;
    ctx.beginPath();
    ctx.arc(center.x, center.y, corePulseR, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(94, 234, 212, 0.35)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Transform and project 3D points
    const cosX = Math.cos(rotX), sinX = Math.sin(rotX);
    const cosY = Math.cos(rotY), sinY = Math.sin(rotY);

    const projected = nodes.map((node) => {
      // Rotate around Y
      let x1 = node.baseX * cosY - node.baseZ * sinY;
      let z1 = node.baseZ * cosY + node.baseX * sinY;

      // Rotate around X
      let y1 = node.baseY * cosX - z1 * sinX;
      let z2 = z1 * cosX + node.baseY * sinX;

      // Perspective projection
      const fov = 400;
      const scale = fov / (fov + z2);
      const px = center.x + x1 * scale;
      const py = center.y + y1 * scale;
      const alpha = (z2 + radius) / (2 * radius); // 0 (back) to 1 (front)

      return {
        x: px,
        y: py,
        z: z2,
        scale,
        alpha: Math.max(0.15, Math.min(1, alpha)),
        size: node.size * scale,
      };
    });

    // Draw connecting neural network lines between close nodes
    for (let i = 0; i < projected.length; i++) {
      for (let j = i + 1; j < projected.length; j++) {
        const p1 = projected[i];
        const p2 = projected[j];
        const dx = p1.x - p2.x;
        const dy = p1.y - p2.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 70) {
          const lineAlpha = (1 - dist / 70) * Math.min(p1.alpha, p2.alpha) * 0.35;
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.strokeStyle = `rgba(56, 189, 248, ${lineAlpha})`;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }

    // Sort projected points so back nodes draw first
    projected.sort((a, b) => a.z - b.z);

    // Draw nodes
    for (const p of projected) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(94, 234, 212, ${p.alpha})`;
      ctx.shadowColor = 'rgba(6, 182, 212, 0.8)';
      ctx.shadowBlur = p.alpha > 0.6 ? 8 : 0;
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // Interactive orbiting satellite particle
    const satAngle = pulseTime * 0.8;
    const satX = center.x + Math.cos(satAngle) * 165;
    const satY = center.y + Math.sin(satAngle) * 65;
    ctx.beginPath();
    ctx.arc(satX, satY, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#10b981';
    ctx.shadowColor = '#10b981';
    ctx.shadowBlur = 12;
    ctx.fill();
    ctx.shadowBlur = 0;

    requestAnimationFrame(render);
  }

  // Start rendering
  render();
})();
