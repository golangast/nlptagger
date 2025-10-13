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


    // --- Dialog Logic (M2 Dialog/Modal) ---
    const dialog = document.getElementById('status-dialog');
    const showDialogBtn = document.getElementById('show-dialog-btn');
    const dialogCancelBtn = document.getElementById('dialog-cancel');

    if (showDialogBtn && dialog && dialogCancelBtn) {
        showDialogBtn.addEventListener('click', () => {
            dialog.classList.add('open');
        });
        dialogCancelBtn.addEventListener('click', () => {
            dialog.classList.remove('open');
        });
        dialog.addEventListener('click', (e) => {
            if (e.target === dialog) { // Close when clicking the scrim (background)
                dialog.classList.remove('open');
            }
        });
    }

    // --- Tab Switching Logic (Segmented Button) ---
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            const targetContent = document.getElementById(`tab-${tabId}`);

            if (button.classList.contains('active')) return;

            // Deactivate old content and tab
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => {
                if (content.classList.contains('active')) {
                    content.style.opacity = '0';
                    content.style.transform = 'translateY(10px)';
                    setTimeout(() => { content.classList.remove('active'); }, 300);
                }
            });

            // Activate new tab button
            button.classList.add('active');

            // Activate new content with a delay
            setTimeout(() => { targetContent.classList.add('active'); }, 150);
        });
    });

    // --- Code Copying & Snackbar Logic ---
    const copyButton = document.querySelector('.copy-btn');
    const codeElement = document.getElementById('go-code-example');
    const snackbar = document.getElementById('m3-snackbar');

    if (copyButton && codeElement && snackbar) {
        copyButton.addEventListener('click', async () => {
            const codeText = codeElement.textContent;
            
            try {
                await navigator.clipboard.writeText(codeText);
                
                // Show Snackbar
                snackbar.classList.remove('show'); // Reset animation
                void snackbar.offsetWidth; // Trigger reflow
                snackbar.classList.add('show');

            } catch (err) {
                console.error('Failed to copy text: ', err);
                alert('Failed to copy code. Please copy manually.');
            }
        });
    }

    // --- Slider Logic ---
    const slider = document.getElementById('batch-slider');
    const sliderValueDisplay = document.querySelector('.slider-value');

    if (slider && sliderValueDisplay) {
        slider.addEventListener('input', () => {
            sliderValueDisplay.textContent = slider.value;
        });
    }
});