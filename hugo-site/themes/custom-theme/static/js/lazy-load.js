document.addEventListener('DOMContentLoaded', function() {
  const images = document.querySelectorAll('.article-card-cover img');
  
  if ('IntersectionObserver' in window) {
    const imageObserver = new IntersectionObserver(function(entries, observer) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          const img = entry.target;
          img.src = img.getAttribute('src');
          img.addEventListener('load', function() {
            img.classList.add('loaded');
          });
          imageObserver.unobserve(img);
        }
      });
    }, {
      rootMargin: '50px 0px',
      threshold: 0.01
    });
    
    images.forEach(function(img) {
      imageObserver.observe(img);
    });
  } else {
    images.forEach(function(img) {
      img.classList.add('loaded');
    });
  }
});
