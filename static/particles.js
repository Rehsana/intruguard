document.addEventListener("DOMContentLoaded", () => {
  const container = document.getElementById("particles");
  if (!container) return;

  const particleCount = 40; // adjust 30-60
  for (let i = 0; i < particleCount; i++) {
    const p = document.createElement("span");
    p.classList.add("particle");

    const size = Math.random() * 4 + 2; // 2px to 6px
    p.style.width = `${size}px`;
    p.style.height = `${size}px`;

    p.style.left = `${Math.random() * 100}%`;
    p.style.animationDuration = `${Math.random() * 10 + 6}s`;
    p.style.animationDelay = `${Math.random() * 6}s`;

    container.appendChild(p);
  }
});
