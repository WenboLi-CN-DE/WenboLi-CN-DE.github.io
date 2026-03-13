// theme-switcher.js
(function() {
  const THEME_KEY = 'blog-theme';
  
  function getPreferredTheme() {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored) return stored;
    
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  
  function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_KEY, theme);
  }
  
  function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    setTheme(next);
  }
  
  // 初始化主题
  setTheme(getPreferredTheme());
  
  // 导出全局函数
  window.toggleTheme = toggleTheme;
})();
