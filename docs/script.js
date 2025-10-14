document.addEventListener('DOMContentLoaded', () => {
    
    // --- Scroll Animation Logic (UNCHANGED) ---
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


    // --- Dialog, Snackbar, and Copy Logic (UNCHANGED) ---
    const dialog = document.getElementById('status-dialog');
    const showDialogBtn = document.getElementById('show-dialog-btn');
    const dialogCancelBtn = document.getElementById('dialog-cancel');
    const viewGithubBtn = dialog ? dialog.querySelector('.custom-button.filled') : null;
    const copyButton = document.querySelector('.copy-btn');
    const codeElement = document.getElementById('go-code-example');
    const snackbar = document.getElementById('custom-snackbar');

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

    // --- Tab Switching Logic (UNCHANGED) ---
    const allTabButtons = document.querySelectorAll('.tab-button');
    
    allTabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            const targetContent = document.getElementById(`tab-${tabId}`);
            const parentContainer = button.closest('.custom-segmented-button');

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


    // --- Navigation Synchronization (Sidebar Link Fix) ---
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
            // Offset by header height (64px) + some margin = ~80px
            const offset = targetElement.offsetTop - 80; 
            contentWrapper.scrollTo({ top: offset, behavior: 'smooth' });
        }
    };
    navItems.forEach(item => item.addEventListener('click', (e) => scrollToSection(e, item)));


    // IntersectionObserver for Nav Active State update
    const sectionObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Ensure the active item is near the top of the viewport to count as 'active'
                if (entry.boundingClientRect.top < contentWrapper.clientHeight / 2) {
                    navItems.forEach(item => item.classList.remove('active'));
                    const activeId = entry.target.id;
                    document.querySelector(`.app-sidebar a[href="#${activeId}"]`)?.classList.add('active');
                }
            }
        });
    }, {
        root: contentWrapper, 
        // Use a small rootMargin to trigger the link when the section enters the top of the viewport
        rootMargin: '-50% 0px 0px 0px', // Adjusted to better handle the top-heavy fixed layout
        threshold: 0.01 // Trigger immediately when even a small part enters
    });

    mainSections.forEach(section => sectionObserver.observe(section));
});

// --- JAVASCRIPT: Copy to Clipboard Feature (Optional) ---
document.addEventListener('DOMContentLoaded', () => {
    const generatedCommandCell = document.querySelector('.generated-command');
    if (generatedCommandCell) {
        // 1. Extract the command text
        const commandText = generatedCommandCell.querySelector('code').innerText.trim();
        
        // 2. Create the Copy Button element
        const copyButton = document.createElement('button');
        copyButton.innerText = '📋 Copy';
        copyButton.className = 'copy-command-btn';
        
        // 3. Add styles to the button
        copyButton.style.cssText = `
            margin-top: 10px;
            padding: 5px 10px;
            background-color: #333;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            transition: background-color 0.2s;
        `;

        // 4. Implement the copy functionality
        copyButton.addEventListener('click', () => {
            navigator.clipboard.writeText(commandText).then(() => {
                copyButton.innerText = '✅ Copied!';
                setTimeout(() => {
                    copyButton.innerText = '📋 Copy';
                }, 2000);
            }).catch(err => {
                console.error('Could not copy text: ', err);
            });
        });

        // 5. Append the button below the command
        generatedCommandCell.appendChild(copyButton);
    }
});