document.addEventListener('DOMContentLoaded', function() {
  const heroBanner = document.getElementById('heroBanner');
  
  if (!heroBanner) {
    return;
  }
  
  let lastScrollTop = 0;
  const scrollThreshold = 100;
  
  function handleScroll() {
    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    
    if (scrollTop > scrollThreshold) {
      heroBanner.classList.add('scrolled');
    } else {
      heroBanner.classList.remove('scrolled');
    }
    
    lastScrollTop = scrollTop;
  }
  
  window.addEventListener('scroll', handleScroll, { passive: true });
  
  handleScroll();
});
