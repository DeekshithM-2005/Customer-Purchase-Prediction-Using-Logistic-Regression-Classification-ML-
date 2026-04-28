/**
 * Customer Purchase Prediction - App JS
 * Mobile nav toggle + Theme switching
 */

document.addEventListener('DOMContentLoaded', function() {
    // --- Mobile Nav Toggle ---
    const navToggle = document.getElementById('navToggle');
    const navRight = document.getElementById('navRight');

    if (navToggle && navRight) {
        navToggle.addEventListener('click', function() {
            navRight.classList.toggle('show');
        });
    }

    // --- Theme Switcher ---
    const themeBtns = document.querySelectorAll('.theme-btn');
    const savedTheme = localStorage.getItem('cpp-theme') || 'skyblue';

    // Apply saved theme on load
    applyTheme(savedTheme);

    themeBtns.forEach(function(btn) {
        btn.addEventListener('click', function() {
            const theme = this.getAttribute('data-theme');
            applyTheme(theme);
            localStorage.setItem('cpp-theme', theme);
        });
    });

    function applyTheme(theme) {
        if (theme === 'black') {
            document.documentElement.setAttribute('data-theme', 'black');
        } else {
            document.documentElement.removeAttribute('data-theme');
        }

        // Update active button
        themeBtns.forEach(function(btn) {
            btn.classList.remove('active');
            if (btn.getAttribute('data-theme') === theme) {
                btn.classList.add('active');
            }
        });
    }
});
