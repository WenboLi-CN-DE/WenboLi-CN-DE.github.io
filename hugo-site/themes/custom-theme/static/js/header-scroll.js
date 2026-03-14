/**
 * Header Scroll Animation
 * When year-header scrolls up to site-header, site-header slides up and out
 */
document.addEventListener('DOMContentLoaded', function() {
  const siteHeader = document.querySelector('.site-header');
  const yearHeaders = document.querySelectorAll('.year-header');
  
  // Only activate on pages with year-headers (archive/list pages)
  if (!siteHeader || yearHeaders.length === 0) {
    return;
  }
  
  let headerHidden = false;
  
  function handleScroll() {
    const siteHeaderBottom = siteHeader.offsetHeight;
    
    // Check if any year-header is touching or overlapping with site-header
    let shouldHideHeader = false;
    
    yearHeaders.forEach(yearHeader => {
      const yearHeaderRect = yearHeader.getBoundingClientRect();
      
      // When year-header's top reaches site-header's bottom
      if (yearHeaderRect.top <= siteHeaderBottom) {
        shouldHideHeader = true;
      }
    });
    
    if (shouldHideHeader && !headerHidden) {
      siteHeader.classList.add('scroll-out');
      headerHidden = true;
    } else if (!shouldHideHeader && headerHidden) {
      siteHeader.classList.remove('scroll-out');
      headerHidden = false;
    }
  }
  
  window.addEventListener('scroll', handleScroll, { passive: true });
  
  // Initial check
  handleScroll();
});
