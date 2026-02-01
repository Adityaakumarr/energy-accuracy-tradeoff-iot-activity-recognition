/**
 * Main JavaScript for Energy-Accuracy Tradeoff Web Application
 * Handles theme toggling, API communication, and interactive features
 */

// ============================================================================
// Theme Management
// ============================================================================

function initTheme() {
    const themeToggle = document.getElementById('themeToggle');
    const currentTheme = localStorage.getItem('theme') || 'light';
    
    document.documentElement.setAttribute('data-theme', currentTheme);
    updateThemeIcon(currentTheme);
    
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const theme = document.documentElement.getAttribute('data-theme');
            const newTheme = theme === 'light' ? 'dark' : 'light';
            
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            updateThemeIcon(newTheme);
        });
    }
}

function updateThemeIcon(theme) {
    const icon = document.querySelector('.theme-icon');
    if (icon) {
        icon.textContent = theme === 'light' ? '🌙' : '☀️';
    }
}

// ============================================================================
// API Helper Functions
// ============================================================================

async function apiGet(endpoint) {
    try {
        const response = await fetch(endpoint);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('API GET error:', error);
        throw error;
    }
}

async function apiPost(endpoint, data) {
    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error('API POST error:', error);
        throw error;
    }
}

// ============================================================================
// Navigation Active State
// ============================================================================

function updateActiveNav() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentPath || (currentPath === '/' && href === '/')) {
            link.style.color = 'var(--primary)';
            link.style.background = 'var(--bg-secondary)';
        }
    });
}

// ============================================================================
// Table Sorting
// ============================================================================

function makeSortable(tableId) {
    const table = document.getElementById(tableId);
    if (!table) return;
    
    const headers = table.querySelectorAll('th');
    headers.forEach((header, index) => {
        header.style.cursor = 'pointer';
        header.addEventListener('click', () => {
            sortTable(table, index);
        });
    });
}

function sortTable(table, columnIndex) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    const sortedRows = rows.sort((a, b) => {
        const aValue = a.cells[columnIndex].textContent.trim();
        const bValue = b.cells[columnIndex].textContent.trim();
        
        // Try to parse as number
        const aNum = parseFloat(aValue.replace(/[^0-9.-]/g, ''));
        const bNum = parseFloat(bValue.replace(/[^0-9.-]/g, ''));
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return aNum - bNum;
        }
        
        return aValue.localeCompare(bValue);
    });
    
    // Clear and re-append sorted rows
    tbody.innerHTML = '';
    sortedRows.forEach(row => tbody.appendChild(row));
}

// ============================================================================
// Notification System
// ============================================================================

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Style the notification
    Object.assign(notification.style, {
        position: 'fixed',
        top: '20px',
        right: '20px',
        padding: '1rem 1.5rem',
        borderRadius: '0.5rem',
        background: type === 'success' ? 'var(--success)' : 
                   type === 'error' ? 'var(--danger)' : 
                   type === 'warning' ? 'var(--warning)' : 'var(--primary)',
        color: 'white',
        fontWeight: '600',
        boxShadow: 'var(--shadow-lg)',
        zIndex: '1000',
        animation: 'slideIn 0.3s ease-out'
    });
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Add animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// ============================================================================
// CSV Parser
// ============================================================================

function parseCSV(text) {
    const lines = text.trim().split('\n');
    const data = lines.map(line => {
        return line.split(',').map(value => parseFloat(value.trim()));
    });
    return data;
}

function validateSensorData(data) {
    if (!Array.isArray(data)) {
        return { valid: false, error: 'Data must be an array' };
    }
    
    if (data.length !== 128) {
        return { valid: false, error: `Expected 128 samples, got ${data.length}` };
    }
    
    for (let i = 0; i < data.length; i++) {
        if (!Array.isArray(data[i]) || data[i].length !== 6) {
            return { valid: false, error: `Row ${i + 1} must have 6 values` };
        }
        
        for (let j = 0; j < data[i].length; j++) {
            if (isNaN(data[i][j])) {
                return { valid: false, error: `Invalid number at row ${i + 1}, column ${j + 1}` };
            }
        }
    }
    
    return { valid: true };
}

// ============================================================================
// Chart Utilities
// ============================================================================

function createBarChart(ctx, labels, data, label, color) {
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                backgroundColor: color,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

function createDoughnutChart(ctx, labels, data, colors) {
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors,
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

// ============================================================================
// Data Formatting
// ============================================================================

function formatNumber(num, decimals = 2) {
    return Number(num).toFixed(decimals);
}

function formatPercent(num, decimals = 1) {
    return (num * 100).toFixed(decimals) + '%';
}

function formatEnergy(uj) {
    if (uj < 1) {
        return (uj * 1000).toFixed(2) + ' nJ';
    }
    return uj.toFixed(2) + ' µJ';
}

// ============================================================================
// Initialize on Page Load
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    updateActiveNav();
    
    // Make results table sortable if it exists
    makeSortable('resultsTable');
    
    console.log('Energy-Accuracy Tradeoff Web App initialized');
});

// ============================================================================
// Export for use in other scripts
// ============================================================================

window.EnergyApp = {
    apiGet,
    apiPost,
    showNotification,
    parseCSV,
    validateSensorData,
    createBarChart,
    createDoughnutChart,
    formatNumber,
    formatPercent,
    formatEnergy
};
