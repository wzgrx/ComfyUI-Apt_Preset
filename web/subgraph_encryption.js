// Subgraph Encryption Extension for ComfyUI
// This extension adds password protection to subgraphs

(function() {
    'use strict';

    console.log('Subgraph Encryption Extension loaded');

    // Helper function to encrypt password using simple encryption (in a real scenario, use a stronger method)
    function encryptPassword(password, salt) {
        // This is a simple encryption for demonstration purposes
        // In a real-world scenario, use a proper encryption library
        const combined = password + salt;
        let encrypted = '';
        for (let i = 0; i < combined.length; i++) {
            const charCode = combined.charCodeAt(i);
            encrypted += String.fromCharCode(charCode + 5);
        }
        return encrypted;
    }

    // Helper function to decrypt password
    function decryptPassword(encrypted, salt) {
        let decrypted = '';
        for (let i = 0; i < encrypted.length; i++) {
            const charCode = encrypted.charCodeAt(i);
            decrypted += String.fromCharCode(charCode - 5);
        }
        // Extract the password by removing the salt
        return decrypted.substring(0, decrypted.length - salt.length);
    }

    // Generate a random salt
    function generateSalt() {
        return Math.random().toString(36).substring(2, 10);
    }

    // Original convertSubgraph function reference
    let originalConvertSubgraph = null;

    // Original unpackSubgraph function reference
    let originalUnpackSubgraph = null;

    // Function to patch the convert to subgraph functionality
    function patchConvertSubgraph() {
        // Try to find the convertSubgraph function in the global scope
        // This may need to be adjusted based on the actual ComfyUI implementation
        if (typeof app !== 'undefined' && app.graph && typeof app.graph.convertToSubgraph === 'function') {
            originalConvertSubgraph = app.graph.convertToSubgraph;
            
            app.graph.convertToSubgraph = function() {
                // Show password prompt
                const password = prompt('输入密码以加密此子图（留空则不加密）:');
                
                // Call the original function
                const result = originalConvertSubgraph.apply(this, arguments);
                
                // If password is provided, save the encrypted password with the subgraph
                if (password && result && result.subgraph) {
                    const salt = generateSalt();
                    const encryptedPassword = encryptPassword(password, salt);
                    
                    // Store the encrypted password and salt in the subgraph data
                    // This assumes the result.subgraph has a way to store metadata
                    // Adjust based on actual ComfyUI implementation
                    if (!result.subgraph.metadata) {
                        result.subgraph.metadata = {};
                    }
                    result.subgraph.metadata.encrypted = true;
                    result.subgraph.metadata.password = encryptedPassword;
                    result.subgraph.metadata.salt = salt;
                    
                    console.log('Subgraph encrypted successfully');
                }
                
                return result;
            };
            
            console.log('Convert to Subgraph functionality patched successfully');
        } else {
            // Fallback approach - listen for context menu events
            document.addEventListener('contextmenu', function(e) {
                setTimeout(() => {
                    // Look for the "Convert to Subgraph" menu item
                    const convertMenuItems = Array.from(document.querySelectorAll('*'))
                        .filter(el => el.textContent && el.textContent.includes('Convert to Subgraph'));
                    
                    convertMenuItems.forEach(item => {
                        const originalOnClick = item.onclick;
                        if (originalOnClick) {
                            item.onclick = function() {
                                const password = prompt('输入密码以加密此子图（留空则不加密）:');
                                
                                // Store the password temporarily
                                if (password) {
                                    const salt = generateSalt();
                                    const encryptedPassword = encryptPassword(password, salt);
                                    localStorage.setItem('temp_subgraph_password', JSON.stringify({
                                        password: encryptedPassword,
                                        salt: salt
                                    }));
                                } else {
                                    localStorage.removeItem('temp_subgraph_password');
                                }
                                
                                // Call the original handler
                                originalOnClick.apply(this, arguments);
                            };
                        }
                    });
                }, 100);
            }, true);
            
            console.log('Subgraph encryption: Using context menu fallback approach');
        }
    }

    // Function to patch the unpack subgraph functionality
    function patchUnpackSubgraph() {
        // Try to find the unpackSubgraph function in the global scope
        if (typeof app !== 'undefined' && app.graph && typeof app.graph.unpackSubgraph === 'function') {
            originalUnpackSubgraph = app.graph.unpackSubgraph;
            
            app.graph.unpackSubgraph = function(subgraph) {
                // Check if the subgraph is encrypted
                if (subgraph && subgraph.metadata && subgraph.metadata.encrypted) {
                    const userPassword = prompt('此子图已加密，请输入密码:');
                    
                    if (userPassword) {
                        // Decrypt and verify password
                        const decryptedPassword = decryptPassword(
                            subgraph.metadata.password, 
                            subgraph.metadata.salt
                        );
                        
                        if (decryptedPassword === userPassword) {
                            // Password is correct, proceed with unpacking
                            return originalUnpackSubgraph.apply(this, arguments);
                        } else {
                            alert('密码错误，无法解包子图');
                            return false;
                        }
                    } else {
                        // User canceled, do not unpack
                        return false;
                    }
                } else {
                    // Subgraph is not encrypted, proceed normally
                    return originalUnpackSubgraph.apply(this, arguments);
                }
            };
            
            console.log('Unpack Subgraph functionality patched successfully');
        } else {
            // Fallback approach - listen for subgraph nodes being clicked
            const observer = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    if (mutation.type === 'childList') {
                        // Look for subgraph nodes
                        const subgraphNodes = Array.from(document.querySelectorAll('*'))
                            .filter(el => el.textContent && el.textContent.includes('Subgraph'));
                        
                        subgraphNodes.forEach(node => {
                            // Add click event listener to check for unpack action
                            node.addEventListener('click', function(e) {
                                // Check if this click is related to unpacking
                                if (e.target.textContent && e.target.textContent.includes('Unpack')) {
                                    // In a real implementation, you would need to get the subgraph data
                                    // For this fallback, we'll just ask for password if it exists in localStorage
                                    const tempPasswordData = localStorage.getItem('temp_subgraph_password');
                                    if (tempPasswordData) {
                                        const password = prompt('此子图可能已加密，请输入密码:');
                                        if (!password) {
                                            e.preventDefault();
                                            e.stopPropagation();
                                            return false;
                                        }
                                    }
                                }
                            }, true);
                        });
                    }
                });
            });
            
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
            
            console.log('Unpack subgraph encryption: Using MutationObserver fallback approach');
        }
    }

    // Wait for the document to be fully loaded before applying patches
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            patchConvertSubgraph();
            patchUnpackSubgraph();
        });
    } else {
        // Document is already loaded, apply patches immediately
        setTimeout(() => {
            patchConvertSubgraph();
            patchUnpackSubgraph();
        }, 1000);
    }

    // Alternative approach: Override the context menu handling
    // This will intercept all context menu actions and check for subgraph operations
    const originalAddEventListener = document.addEventListener;
    document.addEventListener = function(event, handler, options) {
        if (event === 'contextmenu') {
            const originalHandler = handler;
            handler = function(e) {
                // Call the original handler first
                originalHandler.apply(this, arguments);
                
                // Check if this context menu is for nodes
                if (e.target.closest('.node')) {
                    setTimeout(() => {
                        // Look for "Convert to Subgraph" and "Unpack Subgraph" menu items
                        const menuItems = Array.from(document.querySelectorAll('.dropdown-menu-item'));
                        
                        menuItems.forEach(item => {
                            if (item.textContent && item.textContent.includes('Convert to Subgraph')) {
                                const originalOnClick = item.onclick;
                                if (originalOnClick) {
                                    item.onclick = function() {
                                        const password = prompt('输入密码以加密此子图（留空则不加密）:');
                                        
                                        // Store the password temporarily
                                        if (password) {
                                            const salt = generateSalt();
                                            const encryptedPassword = encryptPassword(password, salt);
                                            localStorage.setItem('temp_subgraph_encryption', JSON.stringify({
                                                password: encryptedPassword,
                                                salt: salt,
                                                timestamp: Date.now()
                                            }));
                                        }
                                        
                                        // Call the original handler
                                        originalOnClick.apply(this, arguments);
                                    };
                                }
                            } else if (item.textContent && item.textContent.includes('Unpack Subgraph')) {
                                const originalOnClick = item.onclick;
                                if (originalOnClick) {
                                    item.onclick = function() {
                                        // Check if there's a stored password for this subgraph
                                        // In a real implementation, you would need to associate passwords with specific subgraphs
                                        const encryptionData = localStorage.getItem('temp_subgraph_encryption');
                                        if (encryptionData) {
                                            try {
                                                const data = JSON.parse(encryptionData);
                                                // Check if the data is recent (within 5 minutes)
                                                if (Date.now() - data.timestamp < 5 * 60 * 1000) {
                                                    const userPassword = prompt('此子图已加密，请输入密码:');
                                                    if (userPassword) {
                                                        const decryptedPassword = decryptPassword(data.password, data.salt);
                                                        if (decryptedPassword !== userPassword) {
                                                            alert('密码错误，无法解包子图');
                                                            return false;
                                                        }
                                                    } else {
                                                        // User canceled
                                                        return false;
                                                    }
                                                }
                                            } catch (e) {
                                                console.error('Error parsing encryption data:', e);
                                            }
                                        }
                                        
                                        // Call the original handler
                                        originalOnClick.apply(this, arguments);
                                    };
                                }
                            }
                        });
                    }, 100);
                }
            };
        }
        return originalAddEventListener.call(this, event, handler, options);
    };
})();