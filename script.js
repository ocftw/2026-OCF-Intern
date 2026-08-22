// VLM OCR 實習成果報告：只保留一般瀏覽器 JavaScript，不需要任何 framework。

// 1) 側欄：依目前閱讀位置標示章節。
(() => {
  const links = Array.from(document.querySelectorAll('[data-toc]'));
  const sections = links
    .map(link => document.getElementById(link.getAttribute('data-toc')))
    .filter(Boolean);

  function paint(id) {
    links.forEach(link => {
      link.classList.toggle('is-active', link.dataset.toc === id);
    });
  }

  function onScroll() {
    let current = sections[0]?.id;
    for (const section of sections) {
      if (section.getBoundingClientRect().top <= 140) current = section.id;
    }
    if (current) paint(current);
  }

  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();

// 2) OmniDocBench 雷達圖：座標由資料算出，不在 HTML 手算填死。
//    五個軸的角度是寫死的幾何常數（正五邊形，Text 在正上方、順時針每軸
//    72°），但每個模型的分數只在這裡出現一次；改分數只要改這個陣列。
(() => {
  const RADAR_AXES = ['text', 'teds', 'cdm', 'order', 'overall'];
  const RADAR_CENTER = { x: 170, y: 150 };
  const RADAR_MAX_RADIUS = 110;

  const RADAR_DATA = [
    { model: 'Qwen3-VL 32B',       color: '#2563eb', text: 88.9, teds: 78.5, cdm: 88.5, order: 79.4, overall: 85.3 },
    { model: 'Qwen3-VL 4B',        color: '#3b82f6', text: 85.1, teds: 67.6, cdm: 74.2, order: 75.7, overall: 75.6 },
    { model: 'Gemma 4 31B',        color: '#064e3b', text: 71.4, teds: 63.2, cdm: 81.3, order: 72.4, overall: 71.9 },
    { model: 'InternVL3.5 38B',    color: '#b45309', text: 74.5, teds: 57.9, cdm: 79.0, order: 71.8, overall: 70.5 },
    { model: 'InternVL3.5 4B',     color: '#b45309', text: 80.4, teds: 59.3, cdm: 69.3, order: 74.2, overall: 69.6 },
    { model: 'Gemma 4 26B A4B',    color: '#065f46', text: 62.6, teds: 55.3, cdm: 72.4, order: 64.9, overall: 63.4 },
    { model: 'Gemma 4 12B',        color: '#047857', text: 35.6, teds: 30.1, cdm: 46.7, order: 49.8, overall: 37.5 },
    { model: 'Gemma 4 E4B',        color: '#059669', text: 28.6, teds: 20.4, cdm: 31.7, order: 45.1, overall: 26.9 },
    { model: 'Gemma 4 E2B',        color: '#10b981', text: 19.9, teds: 6.8,  cdm: 25.2, order: 37.3, overall: 17.3 },
  ];

  function radarPoints(entry) {
    return RADAR_AXES.map((key, i) => {
      const angle = (-90 + i * 72) * (Math.PI / 180);
      const r = RADAR_MAX_RADIUS * (entry[key] / 100);
      const x = RADAR_CENTER.x + r * Math.cos(angle);
      const y = RADAR_CENTER.y + r * Math.sin(angle);
      return x.toFixed(1) + ',' + y.toFixed(1);
    }).join(' ');
  }

  function hexToRgba(hex, alpha) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${alpha})`;
  }

  document.querySelectorAll('.radar-series[data-model]').forEach(polygon => {
    const entry = RADAR_DATA.find(d => d.model === polygon.dataset.model);
    if (!entry) return;
    polygon.setAttribute('points', radarPoints(entry));
    polygon.setAttribute('stroke', entry.color);
    polygon.setAttribute('fill', hexToRgba(entry.color, 0.13));
  });
})();

// 3) OmniDocBench 雷達圖模型切換。
(() => {
  const chips = Array.from(document.querySelectorAll('.model-chip'));
  const series = Array.from(document.querySelectorAll('.radar-series'));
  const rows = Array.from(document.querySelectorAll('.radar-row'));

  if (!chips.length) return;

  function rgba(hex, alpha) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${alpha})`;
  }

  function setActive(button, active) {
    const name = button.dataset.model;
    const color = button.dataset.color;
    button.setAttribute('aria-pressed', String(active));
    button.style.background = active ? rgba(color, 0.14) : '#ffffff';
    button.style.borderColor = active ? rgba(color, 0.55) : '#e0e6ed';
    button.style.color = active ? '#101d26' : '#4d5d6b';

    series.filter(el => el.dataset.model === name).forEach(el => {
      el.style.display = active ? '' : 'none';
    });
    rows.filter(el => el.dataset.model === name).forEach(el => {
      el.style.display = active ? '' : 'none';
    });
  }

  chips.forEach(button => {
    // 初始外觀（active/inactive 底色）改由這裡統一計算，不再靠 CSS 另外
    // 定義一組「預設就是 active」的樣式跟這裡的邏輯各自維護一份、容易兩邊不同步。
    setActive(button, button.getAttribute('aria-pressed') === 'true');
    button.addEventListener('click', () => {
      setActive(button, button.getAttribute('aria-pressed') !== 'true');
    });
  });
})();
