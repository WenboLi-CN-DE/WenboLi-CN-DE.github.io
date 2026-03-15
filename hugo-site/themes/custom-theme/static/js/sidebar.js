// Sidebar Toggle Functionality
document.addEventListener('DOMContentLoaded', function() {
    const sidebar = document.querySelector('.sidebar');
    
    if (!sidebar) return;
    
    let toggleButton = null;
    
    function createToggleButton() {
        const button = document.createElement('button');
        button.className = 'sidebar-toggle';
        button.setAttribute('aria-label', 'Toggle Sidebar');
        button.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="3" y1="12" x2="21" y2="12"></line>
                <line x1="3" y1="6" x2="21" y2="6"></line>
                <line x1="3" y1="18" x2="21" y2="18"></line>
            </svg>
        `;
        return button;
    }
    
    function updateButtonIcon(button, isActive) {
        if (isActive) {
            button.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            `;
        } else {
            button.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <line x1="3" y1="12" x2="21" y2="12"></line>
                    <line x1="3" y1="6" x2="21" y2="6"></line>
                    <line x1="3" y1="18" x2="21" y2="18"></line>
                </svg>
            `;
        }
    }
    
    if (window.innerWidth < 1400) {
        toggleButton = createToggleButton();
        document.body.appendChild(toggleButton);
    }
    
    function handleToggleClick(event) {
        event.stopPropagation();
        sidebar.classList.toggle('active');
        document.body.classList.toggle('sidebar-active', sidebar.classList.contains('active'));
        updateButtonIcon(toggleButton, sidebar.classList.contains('active'));
    }
    
    function handleOutsideClick(event) {
        if (toggleButton && !sidebar.contains(event.target) && !toggleButton.contains(event.target)) {
            sidebar.classList.remove('active');
            document.body.classList.remove('sidebar-active');
            if (toggleButton) {
                updateButtonIcon(toggleButton, false);
            }
        }
    }
    
    if (toggleButton) {
        toggleButton.addEventListener('click', handleToggleClick);
        document.addEventListener('click', handleOutsideClick);
    }
    
    window.addEventListener('resize', function() {
        if (window.innerWidth >= 1400) {
            if (toggleButton && toggleButton.parentNode) {
                toggleButton.remove();
                toggleButton = null;
            }
            sidebar.classList.remove('active');
            document.body.classList.remove('sidebar-active');
        } else {
            if (!toggleButton || !document.querySelector('.sidebar-toggle')) {
                toggleButton = createToggleButton();
                document.body.appendChild(toggleButton);
                toggleButton.addEventListener('click', handleToggleClick);
            }
        }
    });
});
