document.addEventListener('DOMContentLoaded', () => {
    
    // --- Scroll Animation Logic (IntersectionObserver) ---
    const targets = document.querySelectorAll('[data-animation-target]');
    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate-in');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });
    targets.forEach(target => { observer.observe(target); });


    // --- Dialog, Snackbar, and Copy Logic (UPDATED IDs) ---
    const dialog = document.getElementById('status-dialog');
    const showDialogBtn = document.getElementById('show-dialog-btn');
    const dialogCancelBtn = document.getElementById('dialog-cancel');
    const viewGithubBtn = dialog ? dialog.querySelector('.custom-button.filled') : null; // Custom button class
    const copyButton = document.querySelector('.copy-btn');
    const codeElement = document.getElementById('go-code-example');
    const snackbar = document.getElementById('custom-snackbar'); // Custom snackbar ID

    if (showDialogBtn && dialog && dialogCancelBtn) {
        showDialogBtn.addEventListener('click', () => { dialog.classList.add('open'); });
        dialogCancelBtn.addEventListener('click', () => { dialog.classList.remove('open'); });
        dialog.addEventListener('click', (e) => {
            if (e.target === dialog) { dialog.classList.remove('open'); }
        });
        if (viewGithubBtn) {
             viewGithubBtn.addEventListener('click', () => {
                window.open('https://github.com/golangast/nlptagger', '_blank');
                dialog.classList.remove('open');
             });
        }
    }
    if (copyButton && codeElement && snackbar) {
        copyButton.addEventListener('click', async () => {
            const codeText = codeElement.textContent;
            try {
                await navigator.clipboard.writeText(codeText);
                snackbar.classList.remove('show');
                void snackbar.offsetWidth;
                snackbar.classList.add('show');
            } catch (err) {
                console.error('Failed to copy text: ', err);
            }
        });
    }

    // --- Tab Switching Logic (UPDATED class selector) ---
    const allTabButtons = document.querySelectorAll('.tab-button');
    
    allTabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            const targetContent = document.getElementById(`tab-${tabId}`);
            const parentContainer = button.closest('.custom-segmented-button'); // Custom class

            if (button.classList.contains('active') || !parentContainer) return;

            parentContainer.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
            
            const tabSection = parentContainer.closest('section');
            if (tabSection) {
                tabSection.querySelectorAll('.tab-content').forEach(content => {
                    if (content.classList.contains('active')) {
                        content.style.opacity = '0';
                        content.style.transform = 'translateY(10px)';
                        setTimeout(() => { content.classList.remove('active'); }, 200);
                    }
                });
            }

            button.classList.add('active');
            setTimeout(() => { if (targetContent) { targetContent.classList.add('active'); } }, 100);
        });
    });


    // --- Navigation Synchronization (Sidebar Link Sync) ---
    const navItems = document.querySelectorAll('.app-sidebar .nav-item');
    const mainSections = document.querySelectorAll('.content-area section');
    const contentWrapper = document.querySelector('.app-content-wrapper');


    // Smooth scroll handler
    const scrollToSection = (e, item) => {
        e.preventDefault();
        const targetId = item.getAttribute('href').substring(1);
        const targetElement = document.getElementById(targetId);
        if (targetElement) {
            // Scroll the app-content-wrapper element
            const offset = targetElement.offsetTop - 80; // Offset by header height
            contentWrapper.scrollTo({ top: offset, behavior: 'smooth' });
        }
    };
    navItems.forEach(item => item.addEventListener('click', (e) => scrollToSection(e, item)));


    // IntersectionObserver for Nav Active State update
    const sectionObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting && entry.intersectionRatio >= 0.5) {
                navItems.forEach(item => item.classList.remove('active'));
                const activeId = entry.target.id;
                document.querySelector(`.app-sidebar a[href="#${activeId}"]`)?.classList.add('active');
            }
        });
    }, {
        threshold: 0.5,
        root: contentWrapper, // Observe relative to the scrolling element
        rootMargin: '-20% 0px -50% 0px'
    });

    mainSections.forEach(section => sectionObserver.observe(section));
});