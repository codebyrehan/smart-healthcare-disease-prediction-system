(() => {
  const orb = document.querySelector('.health-orb');
  if (!orb || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const finePointer = window.matchMedia('(pointer: fine)').matches;
  if (!finePointer) return;
  const hero = document.querySelector('.hero-3d');
  if (!hero) return;
  let raf = 0;
  let tx = 0, ty = 0;
  let cx = 0, cy = 0;
  const render = () => {
    raf = 0;
    cx += (tx - cx) * 0.08;
    cy += (ty - cy) * 0.08;
    orb.style.transform = `translateY(-50%) rotateX(${7 + cy}deg) rotateY(${-12 + cx}deg)`;
    if (Math.abs(tx-cx) > 0.01 || Math.abs(ty-cy) > 0.01) raf = requestAnimationFrame(render);
  };
  hero.addEventListener('pointermove', (event) => {
    const rect = hero.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width - 0.5;
    const y = (event.clientY - rect.top) / rect.height - 0.5;
    tx = x * 10;
    ty = -y * 8;
    if (!raf) raf = requestAnimationFrame(render);
  });
  hero.addEventListener('pointerleave', () => {
    tx = 0; ty = 0;
    if (!raf) raf = requestAnimationFrame(render);
  });
})();
