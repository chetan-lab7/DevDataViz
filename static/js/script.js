// FakeNews Detector JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Handle textarea auto-resize
    const textarea = document.getElementById('news_content');
    if (textarea) {
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    }

    // Animate progress bars on page load
    const progressBars = document.querySelectorAll('.progress-bar');
    if (progressBars.length > 0) {
        progressBars.forEach(bar => {
            const width = bar.style.width;
            bar.style.width = '0%';
            setTimeout(() => {
                bar.style.width = width;
            }, 200);
        });
    }

    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Add fading effect for alerts
    const alerts = document.querySelectorAll('.alert:not(.alert-dismissible)');
    if (alerts.length > 0) {
        alerts.forEach(alert => {
            setTimeout(() => {
                alert.classList.add('fade');
                setTimeout(() => {
                    alert.style.display = 'none';
                }, 500);
            }, 5000);
        });
    }

    // Enable search filter highlight
    const searchQuery = document.querySelector('input[name="query"]')?.value;
    if (searchQuery && searchQuery.length > 0) {
        highlightSearchTerms(searchQuery);
    }
});

// Function to highlight search terms in the table
function highlightSearchTerms(query) {
    if (!query) return;
    
    // Get all table cells that might contain the text
    const cells = document.querySelectorAll('td.text-truncate');
    
    cells.forEach(cell => {
        const content = cell.innerHTML;
        const regex = new RegExp('(' + escapeRegExp(query) + ')', 'gi');
        cell.innerHTML = content.replace(regex, '<mark>$1</mark>');
    });
}

// Helper function to escape special characters in regex
function escapeRegExp(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
