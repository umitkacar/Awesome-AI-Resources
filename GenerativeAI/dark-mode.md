# Dark Mode Implementation: Comprehensive Guide

## Overview

Dark Mode has become an essential feature in modern applications, offering reduced eye strain, improved battery life on OLED displays, and enhanced user experience in low-light conditions. This comprehensive guide covers implementation strategies, best practices, and advanced techniques for creating sophisticated dark mode experiences across web, mobile, and desktop applications.

## Table of Contents

1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Web Implementation](#web-implementation)
4. [React & Modern Frameworks](#react--modern-frameworks)
5. [Mobile Development](#mobile-development)
6. [Design Systems](#design-systems)
7. [Advanced Techniques](#advanced-techniques)
8. [Performance Optimization](#performance-optimization)
9. [Accessibility](#accessibility)
10. [Testing Strategies](#testing-strategies)
11. [Real-world Examples](#real-world-examples)
12. [Best Practices](#best-practices)

## Introduction

Dark mode is more than just inverting colors. It requires thoughtful design decisions, careful color palette selection, and robust implementation strategies to ensure a seamless user experience.

### Benefits of Dark Mode

- **Reduced Eye Strain**: Lower brightness reduces eye fatigue in low-light environments
- **Battery Efficiency**: OLED/AMOLED displays consume less power displaying dark pixels
- **Aesthetic Appeal**: Modern, sleek appearance preferred by many users
- **Accessibility**: Helps users with light sensitivity or certain visual impairments
- **Focus Enhancement**: Reduces distractions and improves content focus

## Core Concepts

### Color Theory for Dark Mode

```javascript
// Color palette structure for dark mode
const colorSystem = {
  light: {
    background: {
      primary: '#FFFFFF',
      secondary: '#F5F5F5',
      tertiary: '#E0E0E0'
    },
    surface: {
      default: '#FFFFFF',
      elevated: '#F8F8F8',
      overlay: 'rgba(0, 0, 0, 0.04)'
    },
    text: {
      primary: 'rgba(0, 0, 0, 0.87)',
      secondary: 'rgba(0, 0, 0, 0.60)',
      disabled: 'rgba(0, 0, 0, 0.38)'
    },
    border: {
      default: '#E0E0E0',
      light: '#F0F0F0',
      dark: '#BDBDBD'
    }
  },
  dark: {
    background: {
      primary: '#121212',
      secondary: '#1E1E1E',
      tertiary: '#2C2C2C'
    },
    surface: {
      default: '#1E1E1E',
      elevated: '#242424',
      overlay: 'rgba(255, 255, 255, 0.04)'
    },
    text: {
      primary: 'rgba(255, 255, 255, 0.87)',
      secondary: 'rgba(255, 255, 255, 0.60)',
      disabled: 'rgba(255, 255, 255, 0.38)'
    },
    border: {
      default: '#383838',
      light: '#2A2A2A',
      dark: '#4A4A4A'
    }
  }
};

// Elevation system for dark mode
const elevationSystem = {
  dark: {
    dp0: '#121212',
    dp1: '#1E1E1E',  // 5% white overlay
    dp2: '#222222',  // 7% white overlay
    dp3: '#242424',  // 8% white overlay
    dp4: '#272727',  // 9% white overlay
    dp6: '#2C2C2C',  // 11% white overlay
    dp8: '#2E2E2E',  // 12% white overlay
    dp12: '#333333', // 14% white overlay
    dp16: '#363636', // 15% white overlay
    dp24: '#383838'  // 16% white overlay
  }
};
```

### Theme Detection and User Preference

```javascript
class ThemeManager {
  constructor() {
    this.theme = this.getInitialTheme();
    this.listeners = new Set();
    this.mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    // Listen for system theme changes
    this.mediaQuery.addEventListener('change', this.handleSystemThemeChange.bind(this));
  }
  
  getInitialTheme() {
    // Priority: User preference > System preference > Default
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      return savedTheme;
    }
    
    if (this.mediaQuery.matches) {
      return 'dark';
    }
    
    return 'light';
  }
  
  handleSystemThemeChange(e) {
    const systemPrefersDark = e.matches;
    const userHasPreference = localStorage.getItem('theme') !== null;
    
    // Only update if user hasn't set a preference
    if (!userHasPreference) {
      this.setTheme(systemPrefersDark ? 'dark' : 'light', false);
    }
  }
  
  setTheme(theme, savePreference = true) {
    this.theme = theme;
    
    if (savePreference) {
      localStorage.setItem('theme', theme);
    }
    
    // Notify all listeners
    this.listeners.forEach(listener => listener(theme));
    
    // Update document
    this.applyTheme(theme);
  }
  
  applyTheme(theme) {
    // Update root element
    document.documentElement.setAttribute('data-theme', theme);
    
    // Update meta theme-color for mobile browsers
    const metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (metaThemeColor) {
      metaThemeColor.content = theme === 'dark' ? '#121212' : '#FFFFFF';
    }
    
    // Dispatch custom event
    window.dispatchEvent(new CustomEvent('themechange', { detail: { theme } }));
  }
  
  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
  
  get currentTheme() {
    return this.theme;
  }
  
  toggle() {
    this.setTheme(this.theme === 'dark' ? 'light' : 'dark');
  }
}

// Usage
const themeManager = new ThemeManager();
```

## Web Implementation

### CSS Custom Properties Approach

```css
/* Base theme variables */
:root {
  /* Light theme (default) */
  --color-bg-primary: #FFFFFF;
  --color-bg-secondary: #F5F5F5;
  --color-bg-tertiary: #E0E0E0;
  
  --color-surface: #FFFFFF;
  --color-surface-elevated: #F8F8F8;
  
  --color-text-primary: rgba(0, 0, 0, 0.87);
  --color-text-secondary: rgba(0, 0, 0, 0.60);
  --color-text-disabled: rgba(0, 0, 0, 0.38);
  
  --color-border: #E0E0E0;
  --color-border-hover: #BDBDBD;
  
  --color-primary: #1976D2;
  --color-primary-dark: #1565C0;
  --color-primary-light: #42A5F5;
  
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
  
  /* Transitions */
  --transition-theme: background-color 0.3s ease, 
                     color 0.3s ease, 
                     border-color 0.3s ease,
                     box-shadow 0.3s ease;
}

/* Dark theme */
[data-theme="dark"] {
  --color-bg-primary: #121212;
  --color-bg-secondary: #1E1E1E;
  --color-bg-tertiary: #2C2C2C;
  
  --color-surface: #1E1E1E;
  --color-surface-elevated: #242424;
  
  --color-text-primary: rgba(255, 255, 255, 0.87);
  --color-text-secondary: rgba(255, 255, 255, 0.60);
  --color-text-disabled: rgba(255, 255, 255, 0.38);
  
  --color-border: #383838;
  --color-border-hover: #4A4A4A;
  
  --color-primary: #90CAF9;
  --color-primary-dark: #64B5F6;
  --color-primary-light: #BBDEFB;
  
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.3);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.5);
}

/* Component styles using variables */
body {
  background-color: var(--color-bg-primary);
  color: var(--color-text-primary);
  transition: var(--transition-theme);
}

.card {
  background-color: var(--color-surface);
  border: 1px solid var(--color-border);
  box-shadow: var(--shadow-md);
  transition: var(--transition-theme);
}

.card:hover {
  border-color: var(--color-border-hover);
  box-shadow: var(--shadow-lg);
}

/* Handle images and media */
[data-theme="dark"] img:not([data-theme-ignore]) {
  opacity: 0.9;
  filter: brightness(0.8) contrast(1.2);
}

[data-theme="dark"] video:not([data-theme-ignore]) {
  opacity: 0.95;
}

/* Syntax highlighting adjustments */
[data-theme="dark"] pre code {
  background-color: #0D1117;
  color: #C9D1D9;
}
```

### JavaScript Theme Switcher

```javascript
class AdvancedThemeSwitcher {
  constructor(options = {}) {
    this.options = {
      storageKey: 'theme-preference',
      defaultTheme: 'light',
      themes: ['light', 'dark'],
      transitionDuration: 300,
      enableSystemDetection: true,
      ...options
    };
    
    this.currentTheme = null;
    this.isTransitioning = false;
    
    this.init();
  }
  
  init() {
    // Get initial theme
    this.currentTheme = this.getStoredTheme() || 
                       this.getSystemTheme() || 
                       this.options.defaultTheme;
    
    // Apply theme immediately
    this.applyTheme(this.currentTheme, false);
    
    // Setup system theme detection
    if (this.options.enableSystemDetection) {
      this.setupSystemThemeDetection();
    }
    
    // Setup keyboard shortcuts
    this.setupKeyboardShortcuts();
    
    // Preload theme assets
    this.preloadThemeAssets();
  }
  
  getStoredTheme() {
    try {
      return localStorage.getItem(this.options.storageKey);
    } catch (e) {
      console.warn('Failed to access localStorage:', e);
      return null;
    }
  }
  
  getSystemTheme() {
    if (!this.options.enableSystemDetection) return null;
    
    const darkModeMediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    return darkModeMediaQuery.matches ? 'dark' : 'light';
  }
  
  setupSystemThemeDetection() {
    const darkModeMediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    darkModeMediaQuery.addEventListener('change', (e) => {
      // Only auto-switch if user hasn't manually set a preference
      if (!this.getStoredTheme()) {
        this.setTheme(e.matches ? 'dark' : 'light');
      }
    });
  }
  
  setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
      // Ctrl/Cmd + Shift + D to toggle dark mode
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'D') {
        e.preventDefault();
        this.toggle();
      }
    });
  }
  
  async applyTheme(theme, animate = true) {
    if (this.isTransitioning) return;
    
    this.isTransitioning = true;
    
    // Start transition
    if (animate) {
      document.documentElement.classList.add('theme-transitioning');
    }
    
    // Update theme attribute
    document.documentElement.setAttribute('data-theme', theme);
    
    // Update meta theme-color
    this.updateMetaThemeColor(theme);
    
    // Load theme-specific assets
    await this.loadThemeAssets(theme);
    
    // Emit event
    this.emitThemeChange(theme);
    
    // End transition
    if (animate) {
      setTimeout(() => {
        document.documentElement.classList.remove('theme-transitioning');
        this.isTransitioning = false;
      }, this.options.transitionDuration);
    } else {
      this.isTransitioning = false;
    }
  }
  
  updateMetaThemeColor(theme) {
    const colors = {
      light: '#FFFFFF',
      dark: '#121212'
    };
    
    let metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (!metaThemeColor) {
      metaThemeColor = document.createElement('meta');
      metaThemeColor.name = 'theme-color';
      document.head.appendChild(metaThemeColor);
    }
    
    metaThemeColor.content = colors[theme] || colors.light;
  }
  
  async loadThemeAssets(theme) {
    // Load theme-specific stylesheets
    const themeStylesheet = document.querySelector(`link[data-theme-stylesheet="${theme}"]`);
    if (themeStylesheet && !themeStylesheet.loaded) {
      await new Promise((resolve) => {
        themeStylesheet.addEventListener('load', resolve, { once: true });
      });
      themeStylesheet.loaded = true;
    }
    
    // Load theme-specific images
    const themeImages = document.querySelectorAll(`img[data-theme-src-${theme}]`);
    const imagePromises = Array.from(themeImages).map(img => {
      return new Promise((resolve) => {
        const newSrc = img.getAttribute(`data-theme-src-${theme}`);
        if (newSrc && img.src !== newSrc) {
          img.addEventListener('load', resolve, { once: true });
          img.src = newSrc;
        } else {
          resolve();
        }
      });
    });
    
    await Promise.all(imagePromises);
  }
  
  preloadThemeAssets() {
    // Preload images for all themes
    this.options.themes.forEach(theme => {
      const images = document.querySelectorAll(`img[data-theme-src-${theme}]`);
      images.forEach(img => {
        const src = img.getAttribute(`data-theme-src-${theme}`);
        if (src) {
          const preloadImg = new Image();
          preloadImg.src = src;
        }
      });
    });
  }
  
  setTheme(theme) {
    if (!this.options.themes.includes(theme)) {
      console.warn(`Theme "${theme}" is not supported`);
      return;
    }
    
    this.currentTheme = theme;
    
    // Store preference
    try {
      localStorage.setItem(this.options.storageKey, theme);
    } catch (e) {
      console.warn('Failed to save theme preference:', e);
    }
    
    // Apply theme
    this.applyTheme(theme);
  }
  
  toggle() {
    const currentIndex = this.options.themes.indexOf(this.currentTheme);
    const nextIndex = (currentIndex + 1) % this.options.themes.length;
    this.setTheme(this.options.themes[nextIndex]);
  }
  
  emitThemeChange(theme) {
    window.dispatchEvent(new CustomEvent('themechange', {
      detail: {
        theme,
        previousTheme: this.currentTheme
      }
    }));
  }
}

// Initialize theme switcher
const themeSwitcher = new AdvancedThemeSwitcher({
  themes: ['light', 'dark', 'auto'],
  transitionDuration: 400
});
```

### Smooth Transitions

```css
/* Theme transition styles */
.theme-transitioning * {
  transition: background-color 0.3s ease,
              color 0.3s ease,
              border-color 0.3s ease,
              fill 0.3s ease,
              stroke 0.3s ease !important;
}

/* Prevent layout shifts during transition */
.theme-transitioning {
  overflow-x: hidden;
}

/* Fade effect for theme switch */
@keyframes theme-fade-in {
  from {
    opacity: 0.8;
  }
  to {
    opacity: 1;
  }
}

[data-theme] {
  animation: theme-fade-in 0.3s ease-out;
}

/* Special handling for images during transition */
.theme-transitioning img {
  transition: opacity 0.3s ease, filter 0.3s ease !important;
}

/* Skeleton screens for loading states */
[data-theme="dark"] .skeleton {
  background: linear-gradient(
    90deg,
    #1E1E1E 0%,
    #2A2A2A 50%,
    #1E1E1E 100%
  );
  background-size: 200% 100%;
  animation: skeleton-loading 1.5s ease-in-out infinite;
}

@keyframes skeleton-loading {
  0% {
    background-position: 200% 0;
  }
  100% {
    background-position: -200% 0;
  }
}
```

## React & Modern Frameworks

### React Context Implementation

```jsx
import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

// Theme context
const ThemeContext = createContext();

// Theme provider component
export const ThemeProvider = ({ children, defaultTheme = 'light' }) => {
  const [theme, setTheme] = useState(() => {
    // Get initial theme from localStorage or system preference
    const saved = localStorage.getItem('theme');
    if (saved) return saved;
    
    if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    
    return defaultTheme;
  });
  
  const [isSystemTheme, setIsSystemTheme] = useState(() => {
    return !localStorage.getItem('theme');
  });
  
  // Apply theme to document
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    
    // Update meta theme-color
    const metaThemeColor = document.querySelector('meta[name="theme-color"]');
    if (metaThemeColor) {
      metaThemeColor.content = theme === 'dark' ? '#121212' : '#FFFFFF';
    }
  }, [theme]);
  
  // Listen for system theme changes
  useEffect(() => {
    if (!isSystemTheme) return;
    
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = (e) => {
      setTheme(e.matches ? 'dark' : 'light');
    };
    
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [isSystemTheme]);
  
  const setThemeWithPreference = useCallback((newTheme) => {
    setTheme(newTheme);
    setIsSystemTheme(false);
    localStorage.setItem('theme', newTheme);
  }, []);
  
  const toggleTheme = useCallback(() => {
    setThemeWithPreference(theme === 'dark' ? 'light' : 'dark');
  }, [theme, setThemeWithPreference]);
  
  const setSystemTheme = useCallback(() => {
    setIsSystemTheme(true);
    localStorage.removeItem('theme');
    
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    setTheme(prefersDark ? 'dark' : 'light');
  }, []);
  
  const value = {
    theme,
    setTheme: setThemeWithPreference,
    toggleTheme,
    setSystemTheme,
    isSystemTheme
  };
  
  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
};

// Hook to use theme
export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

// Theme toggle component
export const ThemeToggle = ({ className }) => {
  const { theme, toggleTheme, setSystemTheme, isSystemTheme } = useTheme();
  const [showMenu, setShowMenu] = useState(false);
  
  return (
    <div className={`theme-toggle ${className}`}>
      <button
        onClick={toggleTheme}
        className="theme-toggle-button"
        aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
      >
        {theme === 'dark' ? '🌙' : '☀️'}
      </button>
      
      <button
        onClick={() => setShowMenu(!showMenu)}
        className="theme-menu-button"
        aria-label="Theme options"
      >
        ⚙️
      </button>
      
      {showMenu && (
        <div className="theme-menu">
          <button onClick={() => setTheme('light')}>
            ☀️ Light
          </button>
          <button onClick={() => setTheme('dark')}>
            🌙 Dark
          </button>
          <button onClick={setSystemTheme}>
            🖥️ System {isSystemTheme && '✓'}
          </button>
        </div>
      )}
    </div>
  );
};
```

### Styled Components Integration

```jsx
import styled, { createGlobalStyle, ThemeProvider } from 'styled-components';

// Theme definitions
const lightTheme = {
  colors: {
    background: {
      primary: '#FFFFFF',
      secondary: '#F5F5F5',
      tertiary: '#E0E0E0'
    },
    text: {
      primary: 'rgba(0, 0, 0, 0.87)',
      secondary: 'rgba(0, 0, 0, 0.60)',
      disabled: 'rgba(0, 0, 0, 0.38)'
    },
    border: {
      default: '#E0E0E0',
      hover: '#BDBDBD'
    },
    primary: '#1976D2',
    error: '#D32F2F',
    warning: '#F57C00',
    success: '#388E3C'
  },
  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.1)'
  }
};

const darkTheme = {
  colors: {
    background: {
      primary: '#121212',
      secondary: '#1E1E1E',
      tertiary: '#2C2C2C'
    },
    text: {
      primary: 'rgba(255, 255, 255, 0.87)',
      secondary: 'rgba(255, 255, 255, 0.60)',
      disabled: 'rgba(255, 255, 255, 0.38)'
    },
    border: {
      default: '#383838',
      hover: '#4A4A4A'
    },
    primary: '#90CAF9',
    error: '#F44336',
    warning: '#FF9800',
    success: '#4CAF50'
  },
  shadows: {
    sm: '0 1px 2px 0 rgba(0, 0, 0, 0.3)',
    md: '0 4px 6px -1px rgba(0, 0, 0, 0.4)',
    lg: '0 10px 15px -3px rgba(0, 0, 0, 0.5)'
  }
};

// Global styles
const GlobalStyle = createGlobalStyle`
  * {
    transition: background-color 0.3s ease, color 0.3s ease;
  }
  
  body {
    background-color: ${props => props.theme.colors.background.primary};
    color: ${props => props.theme.colors.text.primary};
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }
  
  /* Handle images in dark mode */
  ${props => props.theme.mode === 'dark' && `
    img:not([data-no-dark-mode]) {
      opacity: 0.9;
      filter: brightness(0.8) contrast(1.2);
    }
  `}
`;

// Styled components
const Card = styled.div`
  background-color: ${props => props.theme.colors.background.secondary};
  border: 1px solid ${props => props.theme.colors.border.default};
  border-radius: 8px;
  padding: 24px;
  box-shadow: ${props => props.theme.shadows.md};
  
  &:hover {
    border-color: ${props => props.theme.colors.border.hover};
    box-shadow: ${props => props.theme.shadows.lg};
  }
`;

const Button = styled.button`
  background-color: ${props => props.theme.colors.primary};
  color: white;
  border: none;
  border-radius: 4px;
  padding: 8px 16px;
  font-size: 16px;
  cursor: pointer;
  transition: all 0.3s ease;
  
  &:hover {
    opacity: 0.9;
    transform: translateY(-1px);
  }
  
  &:active {
    transform: translateY(0);
  }
`;

// App component
const App = () => {
  const [theme, setTheme] = useState('light');
  
  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  };
  
  const currentTheme = theme === 'light' ? lightTheme : darkTheme;
  
  return (
    <ThemeProvider theme={{ ...currentTheme, mode: theme }}>
      <GlobalStyle />
      <div>
        <Card>
          <h1>Dark Mode with Styled Components</h1>
          <p>This card adapts to the current theme.</p>
          <Button onClick={toggleTheme}>
            Toggle Theme
          </Button>
        </Card>
      </div>
    </ThemeProvider>
  );
};
```

### Next.js Implementation

```jsx
// pages/_app.js
import { useEffect, useState } from 'react';
import { ThemeProvider } from '../contexts/ThemeContext';
import '../styles/globals.css';

function MyApp({ Component, pageProps }) {
  const [mounted, setMounted] = useState(false);
  
  // Prevent flash of incorrect theme
  useEffect(() => {
    setMounted(true);
  }, []);
  
  if (!mounted) {
    return null;
  }
  
  return (
    <ThemeProvider>
      <Component {...pageProps} />
    </ThemeProvider>
  );
}

export default MyApp;

// pages/_document.js
import { Html, Head, Main, NextScript } from 'next/document';

export default function Document() {
  return (
    <Html>
      <Head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                function getInitialTheme() {
                  const persistedTheme = window.localStorage.getItem('theme');
                  if (persistedTheme) {
                    return persistedTheme;
                  }
                  
                  const mql = window.matchMedia('(prefers-color-scheme: dark)');
                  if (mql.matches) {
                    return 'dark';
                  }
                  
                  return 'light';
                }
                
                const theme = getInitialTheme();
                document.documentElement.setAttribute('data-theme', theme);
              })();
            `,
          }}
        />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
```

## Mobile Development

### iOS Implementation

```swift
import UIKit

// MARK: - Theme Manager
class ThemeManager {
    static let shared = ThemeManager()
    
    enum Theme: String, CaseIterable {
        case light
        case dark
        case system
        
        var userInterfaceStyle: UIUserInterfaceStyle {
            switch self {
            case .light:
                return .light
            case .dark:
                return .dark
            case .system:
                return .unspecified
            }
        }
    }
    
    private let themeKey = "app_theme"
    
    var currentTheme: Theme {
        didSet {
            UserDefaults.standard.setValue(currentTheme.rawValue, forKey: themeKey)
            applyTheme()
        }
    }
    
    private init() {
        let savedTheme = UserDefaults.standard.string(forKey: themeKey) ?? Theme.system.rawValue
        self.currentTheme = Theme(rawValue: savedTheme) ?? .system
    }
    
    func applyTheme() {
        guard let windowScene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
              let window = windowScene.windows.first else { return }
        
        UIView.animate(withDuration: 0.3) {
            window.overrideUserInterfaceStyle = self.currentTheme.userInterfaceStyle
        }
    }
}

// MARK: - Custom Colors
extension UIColor {
    static let themed = ThemedColors()
    
    struct ThemedColors {
        let background = UIColor { traitCollection in
            switch traitCollection.userInterfaceStyle {
            case .dark:
                return UIColor(red: 0.07, green: 0.07, blue: 0.07, alpha: 1.0)
            default:
                return .white
            }
        }
        
        let secondaryBackground = UIColor { traitCollection in
            switch traitCollection.userInterfaceStyle {
            case .dark:
                return UIColor(red: 0.12, green: 0.12, blue: 0.12, alpha: 1.0)
            default:
                return UIColor(red: 0.96, green: 0.96, blue: 0.96, alpha: 1.0)
            }
        }
        
        let primaryText = UIColor { traitCollection in
            switch traitCollection.userInterfaceStyle {
            case .dark:
                return UIColor(white: 1.0, alpha: 0.87)
            default:
                return UIColor(white: 0.0, alpha: 0.87)
            }
        }
        
        let secondaryText = UIColor { traitCollection in
            switch traitCollection.userInterfaceStyle {
            case .dark:
                return UIColor(white: 1.0, alpha: 0.60)
            default:
                return UIColor(white: 0.0, alpha: 0.60)
            }
        }
    }
}

// MARK: - Theme-aware View Controller
class ThemedViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()
        setupTheming()
    }
    
    private func setupTheming() {
        view.backgroundColor = .themed.background
    }
    
    override func traitCollectionDidChange(_ previousTraitCollection: UITraitCollection?) {
        super.traitCollectionDidChange(previousTraitCollection)
        
        if traitCollection.hasDifferentColorAppearance(comparedTo: previousTraitCollection) {
            themeDidChange()
        }
    }
    
    func themeDidChange() {
        // Override in subclasses to handle theme changes
    }
}

// MARK: - Custom Theme Toggle
class ThemeToggleView: UIView {
    private let sunIcon = UIImageView(image: UIImage(systemName: "sun.max.fill"))
    private let moonIcon = UIImageView(image: UIImage(systemName: "moon.fill"))
    private let toggleSwitch = UISwitch()
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        setupView()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupView()
    }
    
    private func setupView() {
        // Setup icons
        sunIcon.tintColor = .systemYellow
        moonIcon.tintColor = .systemIndigo
        
        // Setup switch
        toggleSwitch.addTarget(self, action: #selector(toggleChanged), for: .valueChanged)
        
        // Layout
        [sunIcon, toggleSwitch, moonIcon].forEach {
            $0.translatesAutoresizingMaskIntoConstraints = false
            addSubview($0)
        }
        
        NSLayoutConstraint.activate([
            sunIcon.leadingAnchor.constraint(equalTo: leadingAnchor),
            sunIcon.centerYAnchor.constraint(equalTo: centerYAnchor),
            
            toggleSwitch.leadingAnchor.constraint(equalTo: sunIcon.trailingAnchor, constant: 8),
            toggleSwitch.centerYAnchor.constraint(equalTo: centerYAnchor),
            
            moonIcon.leadingAnchor.constraint(equalTo: toggleSwitch.trailingAnchor, constant: 8),
            moonIcon.centerYAnchor.constraint(equalTo: centerYAnchor),
            moonIcon.trailingAnchor.constraint(equalTo: trailingAnchor)
        ])
        
        updateAppearance()
    }
    
    @objc private func toggleChanged() {
        let newTheme: ThemeManager.Theme = toggleSwitch.isOn ? .dark : .light
        ThemeManager.shared.currentTheme = newTheme
        updateAppearance()
    }
    
    private func updateAppearance() {
        let isDark = traitCollection.userInterfaceStyle == .dark
        toggleSwitch.isOn = isDark
        
        UIView.animate(withDuration: 0.3) {
            self.sunIcon.alpha = isDark ? 0.3 : 1.0
            self.moonIcon.alpha = isDark ? 1.0 : 0.3
        }
    }
}
```

### Android Implementation

```kotlin
// ThemeManager.kt
import android.content.Context
import android.content.SharedPreferences
import androidx.appcompat.app.AppCompatDelegate
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

class ThemeManager(context: Context) {
    private val prefs: SharedPreferences = 
        context.getSharedPreferences("theme_prefs", Context.MODE_PRIVATE)
    
    private val _currentTheme = MutableStateFlow(getCurrentTheme())
    val currentTheme: StateFlow<Theme> = _currentTheme
    
    enum class Theme(val value: String) {
        LIGHT("light"),
        DARK("dark"),
        SYSTEM("system");
        
        companion object {
            fun fromValue(value: String): Theme {
                return values().find { it.value == value } ?: SYSTEM
            }
        }
    }
    
    init {
        applyTheme(getCurrentTheme())
    }
    
    fun setTheme(theme: Theme) {
        prefs.edit().putString(THEME_KEY, theme.value).apply()
        _currentTheme.value = theme
        applyTheme(theme)
    }
    
    private fun getCurrentTheme(): Theme {
        val savedTheme = prefs.getString(THEME_KEY, Theme.SYSTEM.value)
        return Theme.fromValue(savedTheme ?: Theme.SYSTEM.value)
    }
    
    private fun applyTheme(theme: Theme) {
        when (theme) {
            Theme.LIGHT -> AppCompatDelegate.setDefaultNightMode(
                AppCompatDelegate.MODE_NIGHT_NO
            )
            Theme.DARK -> AppCompatDelegate.setDefaultNightMode(
                AppCompatDelegate.MODE_NIGHT_YES
            )
            Theme.SYSTEM -> AppCompatDelegate.setDefaultNightMode(
                AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM
            )
        }
    }
    
    companion object {
        private const val THEME_KEY = "app_theme"
        
        @Volatile
        private var INSTANCE: ThemeManager? = null
        
        fun getInstance(context: Context): ThemeManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: ThemeManager(context).also { INSTANCE = it }
            }
        }
    }
}

// themes.xml (res/values)
<resources>
    <!-- Light Theme -->
    <style name="AppTheme.Light" parent="Theme.MaterialComponents.Light.NoActionBar">
        <item name="colorPrimary">@color/primary_light</item>
        <item name="colorPrimaryVariant">@color/primary_variant_light</item>
        <item name="colorSecondary">@color/secondary_light</item>
        <item name="android:colorBackground">@color/background_light</item>
        <item name="colorSurface">@color/surface_light</item>
        <item name="colorError">@color/error_light</item>
        <item name="android:textColorPrimary">@color/text_primary_light</item>
        <item name="android:textColorSecondary">@color/text_secondary_light</item>
    </style>
</resources>

// themes.xml (res/values-night)
<resources>
    <!-- Dark Theme -->
    <style name="AppTheme.Dark" parent="Theme.MaterialComponents.NoActionBar">
        <item name="colorPrimary">@color/primary_dark</item>
        <item name="colorPrimaryVariant">@color/primary_variant_dark</item>
        <item name="colorSecondary">@color/secondary_dark</item>
        <item name="android:colorBackground">@color/background_dark</item>
        <item name="colorSurface">@color/surface_dark</item>
        <item name="colorError">@color/error_dark</item>
        <item name="android:textColorPrimary">@color/text_primary_dark</item>
        <item name="android:textColorSecondary">@color/text_secondary_dark</item>
    </style>
</resources>

// BaseActivity.kt
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch

abstract class BaseActivity : AppCompatActivity() {
    protected lateinit var themeManager: ThemeManager
    
    override fun onCreate(savedInstanceState: Bundle?) {
        themeManager = ThemeManager.getInstance(this)
        super.onCreate(savedInstanceState)
        
        // Observe theme changes
        lifecycleScope.launch {
            themeManager.currentTheme.collect { theme ->
                // Handle theme change if needed
                onThemeChanged(theme)
            }
        }
    }
    
    open fun onThemeChanged(theme: ThemeManager.Theme) {
        // Override in subclasses if needed
    }
}

// Custom View with theme support
class ThemedCardView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : MaterialCardView(context, attrs, defStyleAttr) {
    
    init {
        setupTheming()
    }
    
    private fun setupTheming() {
        val typedArray = context.obtainStyledAttributes(
            intArrayOf(
                android.R.attr.colorBackground,
                com.google.android.material.R.attr.colorSurface
            )
        )
        
        val backgroundColor = typedArray.getColor(1, Color.WHITE)
        typedArray.recycle()
        
        setCardBackgroundColor(backgroundColor)
        cardElevation = resources.getDimension(R.dimen.card_elevation)
        radius = resources.getDimension(R.dimen.card_corner_radius)
    }
}
```

### React Native Implementation

```jsx
// ThemeContext.js
import React, { createContext, useContext, useState, useEffect } from 'react';
import { Appearance, useColorScheme } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';

const ThemeContext = createContext();

const lightTheme = {
  dark: false,
  colors: {
    primary: '#1976D2',
    background: '#FFFFFF',
    card: '#F5F5F5',
    text: 'rgba(0, 0, 0, 0.87)',
    textSecondary: 'rgba(0, 0, 0, 0.60)',
    border: '#E0E0E0',
    notification: '#FF3D00',
  }
};

const darkTheme = {
  dark: true,
  colors: {
    primary: '#90CAF9',
    background: '#121212',
    card: '#1E1E1E',
    text: 'rgba(255, 255, 255, 0.87)',
    textSecondary: 'rgba(255, 255, 255, 0.60)',
    border: '#383838',
    notification: '#FF6E40',
  }
};

export const ThemeProvider = ({ children }) => {
  const systemColorScheme = useColorScheme();
  const [theme, setTheme] = useState(null);
  const [isSystemTheme, setIsSystemTheme] = useState(true);
  
  useEffect(() => {
    loadThemePreference();
  }, []);
  
  useEffect(() => {
    if (isSystemTheme) {
      setTheme(systemColorScheme === 'dark' ? darkTheme : lightTheme);
    }
  }, [systemColorScheme, isSystemTheme]);
  
  const loadThemePreference = async () => {
    try {
      const savedTheme = await AsyncStorage.getItem('theme_preference');
      if (savedTheme) {
        setIsSystemTheme(false);
        setTheme(savedTheme === 'dark' ? darkTheme : lightTheme);
      } else {
        setTheme(systemColorScheme === 'dark' ? darkTheme : lightTheme);
      }
    } catch (error) {
      console.error('Error loading theme preference:', error);
    }
  };
  
  const toggleTheme = async () => {
    const newTheme = theme.dark ? lightTheme : darkTheme;
    setTheme(newTheme);
    setIsSystemTheme(false);
    
    try {
      await AsyncStorage.setItem('theme_preference', newTheme.dark ? 'dark' : 'light');
    } catch (error) {
      console.error('Error saving theme preference:', error);
    }
  };
  
  const setSystemTheme = async () => {
    setIsSystemTheme(true);
    try {
      await AsyncStorage.removeItem('theme_preference');
    } catch (error) {
      console.error('Error removing theme preference:', error);
    }
  };
  
  if (!theme) {
    return null; // Or a loading component
  }
  
  return (
    <ThemeContext.Provider value={{
      theme,
      toggleTheme,
      setSystemTheme,
      isSystemTheme
    }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

// ThemedComponents.js
import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useTheme } from './ThemeContext';

export const ThemedView = ({ style, ...props }) => {
  const { theme } = useTheme();
  
  return (
    <View 
      style={[
        { backgroundColor: theme.colors.background },
        style
      ]} 
      {...props} 
    />
  );
};

export const ThemedText = ({ style, secondary, ...props }) => {
  const { theme } = useTheme();
  
  return (
    <Text 
      style={[
        { 
          color: secondary ? theme.colors.textSecondary : theme.colors.text 
        },
        style
      ]} 
      {...props} 
    />
  );
};

export const ThemedCard = ({ style, children, ...props }) => {
  const { theme } = useTheme();
  
  return (
    <View 
      style={[
        styles.card,
        {
          backgroundColor: theme.colors.card,
          borderColor: theme.colors.border,
          shadowColor: theme.dark ? '#000' : '#000',
          shadowOpacity: theme.dark ? 0.3 : 0.1,
        },
        style
      ]}
      {...props}
    >
      {children}
    </View>
  );
};

export const ThemeToggle = () => {
  const { theme, toggleTheme, setSystemTheme, isSystemTheme } = useTheme();
  
  return (
    <View style={styles.toggleContainer}>
      <TouchableOpacity 
        onPress={toggleTheme}
        style={[
          styles.toggleButton,
          { backgroundColor: theme.colors.primary }
        ]}
      >
        <Text style={styles.toggleText}>
          {theme.dark ? '🌙' : '☀️'} {theme.dark ? 'Dark' : 'Light'}
        </Text>
      </TouchableOpacity>
      
      <TouchableOpacity 
        onPress={setSystemTheme}
        style={[
          styles.systemButton,
          { 
            backgroundColor: isSystemTheme ? theme.colors.primary : theme.colors.card,
            borderColor: theme.colors.border
          }
        ]}
      >
        <Text style={[
          styles.toggleText,
          { color: isSystemTheme ? '#FFFFFF' : theme.colors.text }
        ]}>
          System
        </Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    borderRadius: 8,
    padding: 16,
    marginVertical: 8,
    marginHorizontal: 16,
    borderWidth: 1,
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowRadius: 4,
    elevation: 3,
  },
  toggleContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 16,
  },
  toggleButton: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
    marginRight: 10,
  },
  systemButton: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
    borderWidth: 1,
  },
  toggleText: {
    color: '#FFFFFF',
    fontWeight: 'bold',
  },
});
```

## Design Systems

### Comprehensive Color System

```javascript
// colorSystem.js
export const createColorSystem = () => {
  // Base colors
  const baseColors = {
    gray: {
      50: '#FAFAFA',
      100: '#F5F5F5',
      200: '#E0E0E0',
      300: '#C0C0C0',
      400: '#A0A0A0',
      500: '#808080',
      600: '#606060',
      700: '#404040',
      800: '#202020',
      900: '#101010',
    },
    blue: {
      50: '#E3F2FD',
      100: '#BBDEFB',
      200: '#90CAF9',
      300: '#64B5F6',
      400: '#42A5F5',
      500: '#2196F3',
      600: '#1E88E5',
      700: '#1976D2',
      800: '#1565C0',
      900: '#0D47A1',
    },
    // Add more color scales as needed
  };
  
  // Semantic colors for light theme
  const lightColors = {
    background: {
      default: baseColors.gray[50],
      paper: '#FFFFFF',
      inverse: baseColors.gray[900],
    },
    surface: {
      default: '#FFFFFF',
      raised: baseColors.gray[50],
      overlay: 'rgba(0, 0, 0, 0.04)',
    },
    text: {
      primary: 'rgba(0, 0, 0, 0.87)',
      secondary: 'rgba(0, 0, 0, 0.60)',
      disabled: 'rgba(0, 0, 0, 0.38)',
      inverse: 'rgba(255, 255, 255, 0.87)',
    },
    divider: 'rgba(0, 0, 0, 0.12)',
    action: {
      active: 'rgba(0, 0, 0, 0.54)',
      hover: 'rgba(0, 0, 0, 0.04)',
      selected: 'rgba(0, 0, 0, 0.08)',
      disabled: 'rgba(0, 0, 0, 0.26)',
      disabledBackground: 'rgba(0, 0, 0, 0.12)',
    },
    primary: {
      main: baseColors.blue[700],
      light: baseColors.blue[500],
      dark: baseColors.blue[900],
      contrastText: '#FFFFFF',
    },
  };
  
  // Semantic colors for dark theme
  const darkColors = {
    background: {
      default: '#121212',
      paper: baseColors.gray[800],
      inverse: baseColors.gray[50],
    },
    surface: {
      default: baseColors.gray[800],
      raised: baseColors.gray[700],
      overlay: 'rgba(255, 255, 255, 0.04)',
    },
    text: {
      primary: 'rgba(255, 255, 255, 0.87)',
      secondary: 'rgba(255, 255, 255, 0.60)',
      disabled: 'rgba(255, 255, 255, 0.38)',
      inverse: 'rgba(0, 0, 0, 0.87)',
    },
    divider: 'rgba(255, 255, 255, 0.12)',
    action: {
      active: 'rgba(255, 255, 255, 0.54)',
      hover: 'rgba(255, 255, 255, 0.04)',
      selected: 'rgba(255, 255, 255, 0.08)',
      disabled: 'rgba(255, 255, 255, 0.26)',
      disabledBackground: 'rgba(255, 255, 255, 0.12)',
    },
    primary: {
      main: baseColors.blue[200],
      light: baseColors.blue[100],
      dark: baseColors.blue[300],
      contrastText: 'rgba(0, 0, 0, 0.87)',
    },
  };
  
  return {
    light: lightColors,
    dark: darkColors,
    base: baseColors,
  };
};

// Typography system
export const createTypographySystem = (theme) => {
  const fontFamily = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  
  return {
    fontFamily,
    h1: {
      fontFamily,
      fontWeight: 300,
      fontSize: '96px',
      lineHeight: 1.167,
      letterSpacing: '-1.5px',
      color: theme.text.primary,
    },
    h2: {
      fontFamily,
      fontWeight: 300,
      fontSize: '60px',
      lineHeight: 1.2,
      letterSpacing: '-0.5px',
      color: theme.text.primary,
    },
    h3: {
      fontFamily,
      fontWeight: 400,
      fontSize: '48px',
      lineHeight: 1.167,
      letterSpacing: '0px',
      color: theme.text.primary,
    },
    h4: {
      fontFamily,
      fontWeight: 400,
      fontSize: '34px',
      lineHeight: 1.235,
      letterSpacing: '0.25px',
      color: theme.text.primary,
    },
    h5: {
      fontFamily,
      fontWeight: 400,
      fontSize: '24px',
      lineHeight: 1.334,
      letterSpacing: '0px',
      color: theme.text.primary,
    },
    h6: {
      fontFamily,
      fontWeight: 500,
      fontSize: '20px',
      lineHeight: 1.6,
      letterSpacing: '0.15px',
      color: theme.text.primary,
    },
    body1: {
      fontFamily,
      fontWeight: 400,
      fontSize: '16px',
      lineHeight: 1.5,
      letterSpacing: '0.5px',
      color: theme.text.primary,
    },
    body2: {
      fontFamily,
      fontWeight: 400,
      fontSize: '14px',
      lineHeight: 1.43,
      letterSpacing: '0.25px',
      color: theme.text.secondary,
    },
    button: {
      fontFamily,
      fontWeight: 500,
      fontSize: '14px',
      lineHeight: 1.75,
      letterSpacing: '1.25px',
      textTransform: 'uppercase',
      color: theme.text.primary,
    },
    caption: {
      fontFamily,
      fontWeight: 400,
      fontSize: '12px',
      lineHeight: 1.66,
      letterSpacing: '0.4px',
      color: theme.text.secondary,
    },
    overline: {
      fontFamily,
      fontWeight: 400,
      fontSize: '10px',
      lineHeight: 2.66,
      letterSpacing: '1.5px',
      textTransform: 'uppercase',
      color: theme.text.secondary,
    },
  };
};

// Spacing system
export const spacing = {
  unit: 8,
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
};

// Breakpoints
export const breakpoints = {
  xs: 0,
  sm: 600,
  md: 960,
  lg: 1280,
  xl: 1920,
};

// Shadows for both themes
export const createShadows = (isDark) => {
  const shadowColor = isDark ? 'rgba(0, 0, 0, 0.8)' : 'rgba(0, 0, 0, 0.2)';
  const ambientColor = isDark ? 'rgba(0, 0, 0, 0.6)' : 'rgba(0, 0, 0, 0.14)';
  
  return {
    0: 'none',
    1: `0px 2px 1px -1px ${shadowColor}, 0px 1px 1px 0px ${ambientColor}`,
    2: `0px 3px 1px -2px ${shadowColor}, 0px 2px 2px 0px ${ambientColor}`,
    3: `0px 3px 3px -2px ${shadowColor}, 0px 3px 4px 0px ${ambientColor}`,
    4: `0px 2px 4px -1px ${shadowColor}, 0px 4px 5px 0px ${ambientColor}`,
    6: `0px 3px 5px -1px ${shadowColor}, 0px 6px 10px 0px ${ambientColor}`,
    8: `0px 5px 5px -3px ${shadowColor}, 0px 8px 10px 1px ${ambientColor}`,
    12: `0px 7px 8px -4px ${shadowColor}, 0px 12px 17px 2px ${ambientColor}`,
    16: `0px 8px 10px -5px ${shadowColor}, 0px 16px 24px 2px ${ambientColor}`,
    24: `0px 11px 15px -7px ${shadowColor}, 0px 24px 38px 3px ${ambientColor}`,
  };
};
```

## Advanced Techniques

### Smooth Theme Transitions with FLIP

```javascript
// FLIP (First, Last, Invert, Play) animation for theme transitions
class FLIPThemeTransition {
  constructor(options = {}) {
    this.duration = options.duration || 600;
    this.easing = options.easing || 'cubic-bezier(0.4, 0, 0.2, 1)';
    this.elements = new Map();
  }
  
  captureState(selector = '*') {
    const elements = document.querySelectorAll(selector);
    
    elements.forEach(element => {
      const rect = element.getBoundingClientRect();
      const computed = window.getComputedStyle(element);
      
      this.elements.set(element, {
        first: {
          x: rect.left,
          y: rect.top,
          width: rect.width,
          height: rect.height,
          backgroundColor: computed.backgroundColor,
          color: computed.color,
          opacity: computed.opacity,
        }
      });
    });
  }
  
  transition(themeChangeCallback) {
    // Capture first state
    this.captureState();
    
    // Execute theme change
    themeChangeCallback();
    
    // Capture last state and animate
    this.elements.forEach((state, element) => {
      const last = element.getBoundingClientRect();
      const lastComputed = window.getComputedStyle(element);
      
      // Calculate deltas
      const deltaX = state.first.x - last.left;
      const deltaY = state.first.y - last.top;
      const deltaW = state.first.width / last.width;
      const deltaH = state.first.height / last.height;
      
      // Check if colors changed
      const colorChanged = state.first.backgroundColor !== lastComputed.backgroundColor ||
                          state.first.color !== lastComputed.color;
      
      if (deltaX || deltaY || deltaW !== 1 || deltaH !== 1 || colorChanged) {
        // Apply inverted state
        element.style.transform = `translate(${deltaX}px, ${deltaY}px) scale(${deltaW}, ${deltaH})`;
        element.style.transformOrigin = 'top left';
        
        if (colorChanged) {
          element.style.transition = 'none';
          
          // Force color transition
          element.animate([
            {
              backgroundColor: state.first.backgroundColor,
              color: state.first.color,
            },
            {
              backgroundColor: lastComputed.backgroundColor,
              color: lastComputed.color,
            }
          ], {
            duration: this.duration,
            easing: this.easing,
            fill: 'both'
          });
        }
        
        // Play animation
        element.animate([
          {
            transform: `translate(${deltaX}px, ${deltaY}px) scale(${deltaW}, ${deltaH})`,
          },
          {
            transform: 'none',
          }
        ], {
          duration: this.duration,
          easing: this.easing,
        }).onfinish = () => {
          element.style.transform = '';
          element.style.transformOrigin = '';
          element.style.transition = '';
        };
      }
    });
    
    // Clear stored elements
    this.elements.clear();
  }
}

// Usage
const flipTransition = new FLIPThemeTransition({ duration: 800 });

function changeTheme(newTheme) {
  flipTransition.transition(() => {
    document.documentElement.setAttribute('data-theme', newTheme);
  });
}
```

### Intelligent Image Adaptation

```javascript
class SmartImageAdapter {
  constructor() {
    this.cache = new Map();
    this.observer = null;
    this.init();
  }
  
  init() {
    // Setup Intersection Observer for lazy processing
    this.observer = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            this.processImage(entry.target);
          }
        });
      },
      { rootMargin: '50px' }
    );
    
    // Listen for theme changes
    window.addEventListener('themechange', this.handleThemeChange.bind(this));
  }
  
  observe(image) {
    if (image.dataset.darkModeAdapt !== 'false') {
      this.observer.observe(image);
    }
  }
  
  async processImage(img) {
    const src = img.src;
    const isDarkTheme = document.documentElement.getAttribute('data-theme') === 'dark';
    
    // Check if we have a dark variant
    const darkSrc = img.dataset.darkSrc;
    if (darkSrc && isDarkTheme) {
      this.smoothSwapImage(img, darkSrc);
      return;
    }
    
    // Apply smart filters for dark mode
    if (isDarkTheme && !img.dataset.darkModeProcessed) {
      await this.applySmartDarkModeFilter(img);
    }
  }
  
  async applySmartDarkModeFilter(img) {
    // Analyze image characteristics
    const analysis = await this.analyzeImage(img);
    
    // Apply appropriate filters based on analysis
    const filters = this.calculateFilters(analysis);
    
    img.style.filter = filters;
    img.dataset.darkModeProcessed = 'true';
  }
  
  async analyzeImage(img) {
    const cacheKey = img.src;
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }
    
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Sample image at lower resolution for performance
    const sampleSize = 100;
    canvas.width = sampleSize;
    canvas.height = sampleSize;
    
    await new Promise(resolve => {
      const tempImg = new Image();
      tempImg.onload = () => {
        ctx.drawImage(tempImg, 0, 0, sampleSize, sampleSize);
        resolve();
      };
      tempImg.src = img.src;
    });
    
    const imageData = ctx.getImageData(0, 0, sampleSize, sampleSize);
    const data = imageData.data;
    
    let brightness = 0;
    let saturation = 0;
    let whitePixels = 0;
    
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      
      // Calculate brightness
      brightness += (r + g + b) / 3;
      
      // Calculate saturation
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      saturation += max - min;
      
      // Count white/light pixels
      if (r > 240 && g > 240 && b > 240) {
        whitePixels++;
      }
    }
    
    const pixelCount = data.length / 4;
    const analysis = {
      avgBrightness: brightness / pixelCount / 255,
      avgSaturation: saturation / pixelCount / 255,
      whitePercentage: whitePixels / pixelCount,
      isLogo: this.detectIfLogo(img),
      isPhoto: this.detectIfPhoto(img),
    };
    
    this.cache.set(cacheKey, analysis);
    return analysis;
  }
  
  calculateFilters(analysis) {
    const filters = [];
    
    // Brightness adjustment
    if (analysis.avgBrightness > 0.7) {
      filters.push(`brightness(${0.8 - analysis.whitePercentage * 0.1})`);
    }
    
    // Contrast adjustment for photos
    if (analysis.isPhoto) {
      filters.push('contrast(1.1)');
    }
    
    // Invert logos with high white percentage
    if (analysis.isLogo && analysis.whitePercentage > 0.7) {
      filters.push('invert(0.85)');
      filters.push('hue-rotate(180deg)');
    }
    
    // Reduce saturation slightly for better dark mode appearance
    if (analysis.avgSaturation > 0.5) {
      filters.push('saturate(0.9)');
    }
    
    return filters.join(' ');
  }
  
  detectIfLogo(img) {
    // Simple heuristic: logos are typically small and square-ish
    const aspectRatio = img.naturalWidth / img.naturalHeight;
    const isSmall = img.naturalWidth < 200 || img.naturalHeight < 200;
    const isSquarish = aspectRatio > 0.7 && aspectRatio < 1.3;
    
    return isSmall && isSquarish;
  }
  
  detectIfPhoto(img) {
    // Photos typically have JPEG extension and larger dimensions
    const isJpeg = /\.(jpg|jpeg)$/i.test(img.src);
    const isLarge = img.naturalWidth > 400 && img.naturalHeight > 400;
    
    return isJpeg && isLarge;
  }
  
  smoothSwapImage(img, newSrc) {
    // Preload new image
    const newImg = new Image();
    newImg.onload = () => {
      // Fade transition
      img.style.transition = 'opacity 0.3s ease';
      img.style.opacity = '0';
      
      setTimeout(() => {
        img.src = newSrc;
        img.style.opacity = '1';
      }, 300);
    };
    newImg.src = newSrc;
  }
  
  handleThemeChange(event) {
    const isDark = event.detail.theme === 'dark';
    
    // Process all observed images
    document.querySelectorAll('img[data-dark-mode-processed]').forEach(img => {
      if (isDark) {
        this.applySmartDarkModeFilter(img);
      } else {
        img.style.filter = '';
        img.dataset.darkModeProcessed = '';
      }
    });
    
    // Handle images with dark variants
    document.querySelectorAll('img[data-dark-src]').forEach(img => {
      const originalSrc = img.dataset.originalSrc || img.src;
      const darkSrc = img.dataset.darkSrc;
      
      if (!img.dataset.originalSrc) {
        img.dataset.originalSrc = img.src;
      }
      
      this.smoothSwapImage(img, isDark ? darkSrc : originalSrc);
    });
  }
}

// Initialize
const imageAdapter = new SmartImageAdapter();

// Auto-observe all images
document.querySelectorAll('img').forEach(img => {
  imageAdapter.observe(img);
});

// Observe new images
const imgObserver = new MutationObserver(mutations => {
  mutations.forEach(mutation => {
    mutation.addedNodes.forEach(node => {
      if (node.tagName === 'IMG') {
        imageAdapter.observe(node);
      }
      if (node.querySelectorAll) {
        node.querySelectorAll('img').forEach(img => {
          imageAdapter.observe(img);
        });
      }
    });
  });
});

imgObserver.observe(document.body, {
  childList: true,
  subtree: true
});
```

## Performance Optimization

### CSS-Only Dark Mode

```css
/* Optimized CSS-only dark mode using modern CSS features */
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
  }
  
  /* Use CSS custom properties for performance */
  html {
    --bg: #121212;
    --surface: #1E1E1E;
    --text: rgba(255, 255, 255, 0.87);
    --text-secondary: rgba(255, 255, 255, 0.60);
  }
}

@media (prefers-color-scheme: light) {
  :root {
    color-scheme: light;
  }
  
  html {
    --bg: #FFFFFF;
    --surface: #F5F5F5;
    --text: rgba(0, 0, 0, 0.87);
    --text-secondary: rgba(0, 0, 0, 0.60);
  }
}

/* Use logical properties for better performance */
body {
  background-color: var(--bg);
  color: var(--text);
  
  /* Optimize repaints */
  will-change: background-color, color;
  
  /* Hardware acceleration */
  transform: translateZ(0);
}

/* Efficient selector grouping */
.card,
.panel,
.modal {
  background-color: var(--surface);
  
  /* Single declaration for multiple properties */
  transition: 
    background-color 0.3s ease,
    transform 0.2s ease,
    box-shadow 0.2s ease;
}

/* Use contain property for performance */
.theme-container {
  contain: style layout paint;
}

/* Optimize animations */
@media (prefers-reduced-motion: no-preference) {
  .theme-transition {
    transition: all 0.3s ease;
  }
}

@media (prefers-reduced-motion: reduce) {
  .theme-transition {
    transition: none;
  }
}
```

### JavaScript Performance Optimization

```javascript
// Optimized theme manager with performance considerations
class PerformantThemeManager {
  constructor() {
    // Use WeakMap for better memory management
    this.elementCache = new WeakMap();
    this.rafId = null;
    this.pendingUpdates = new Set();
    
    // Debounce theme changes
    this.debouncedApplyTheme = this.debounce(this.applyTheme.bind(this), 100);
  }
  
  setTheme(theme) {
    // Cancel any pending updates
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
    }
    
    // Store theme
    localStorage.setItem('theme', theme);
    
    // Use requestAnimationFrame for smooth updates
    this.rafId = requestAnimationFrame(() => {
      this.debouncedApplyTheme(theme);
    });
  }
  
  applyTheme(theme) {
    // Batch DOM updates
    const updates = [];
    
    // Update root element
    updates.push(() => {
      document.documentElement.setAttribute('data-theme', theme);
    });
    
    // Update meta theme-color
    updates.push(() => {
      const meta = document.querySelector('meta[name="theme-color"]');
      if (meta) {
        meta.content = theme === 'dark' ? '#121212' : '#FFFFFF';
      }
    });
    
    // Process pending element updates
    this.pendingUpdates.forEach(element => {
      updates.push(() => this.updateElement(element, theme));
    });
    
    // Execute all updates in a single frame
    requestAnimationFrame(() => {
      updates.forEach(update => update());
      this.pendingUpdates.clear();
    });
    
    // Emit event
    this.emitThemeChange(theme);
  }
  
  updateElement(element, theme) {
    // Use cached styles for performance
    let styles = this.elementCache.get(element);
    
    if (!styles) {
      styles = {
        light: element.getAttribute('data-light-style'),
        dark: element.getAttribute('data-dark-style')
      };
      this.elementCache.set(element, styles);
    }
    
    if (styles[theme]) {
      element.style.cssText = styles[theme];
    }
  }
  
  observeElement(element) {
    // Use Intersection Observer for viewport-based updates
    if (!this.observer) {
      this.observer = new IntersectionObserver(
        entries => {
          entries.forEach(entry => {
            if (entry.isIntersecting) {
              this.pendingUpdates.add(entry.target);
            }
          });
        },
        { rootMargin: '100px' }
      );
    }
    
    this.observer.observe(element);
  }
  
  debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }
  
  emitThemeChange(theme) {
    // Use CustomEvent for better performance
    const event = new CustomEvent('themechange', {
      detail: { theme },
      bubbles: true,
      composed: true
    });
    
    document.dispatchEvent(event);
  }
  
  // Cleanup method
  destroy() {
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
    }
    
    if (this.observer) {
      this.observer.disconnect();
    }
    
    this.elementCache = new WeakMap();
    this.pendingUpdates.clear();
  }
}

// Lazy load theme-specific resources
class ThemeResourceLoader {
  constructor() {
    this.loadedResources = new Set();
  }
  
  async loadThemeResources(theme) {
    const resources = this.getThemeResources(theme);
    
    // Load resources in parallel
    const promises = resources.map(resource => {
      if (this.loadedResources.has(resource.url)) {
        return Promise.resolve();
      }
      
      return this.loadResource(resource);
    });
    
    await Promise.all(promises);
  }
  
  getThemeResources(theme) {
    const resources = [];
    
    // Theme-specific stylesheets
    if (theme === 'dark') {
      resources.push({
        type: 'stylesheet',
        url: '/css/dark-theme-extras.css'
      });
    }
    
    // Theme-specific fonts
    resources.push({
      type: 'font',
      url: `/fonts/${theme}-theme-icons.woff2`
    });
    
    return resources;
  }
  
  async loadResource(resource) {
    switch (resource.type) {
      case 'stylesheet':
        await this.loadStylesheet(resource.url);
        break;
      case 'font':
        await this.loadFont(resource.url);
        break;
    }
    
    this.loadedResources.add(resource.url);
  }
  
  loadStylesheet(url) {
    return new Promise((resolve, reject) => {
      const link = document.createElement('link');
      link.rel = 'stylesheet';
      link.href = url;
      link.onload = resolve;
      link.onerror = reject;
      document.head.appendChild(link);
    });
  }
  
  loadFont(url) {
    return new Promise((resolve, reject) => {
      const link = document.createElement('link');
      link.rel = 'preload';
      link.as = 'font';
      link.type = 'font/woff2';
      link.href = url;
      link.crossOrigin = 'anonymous';
      link.onload = resolve;
      link.onerror = reject;
      document.head.appendChild(link);
    });
  }
}
```

## Accessibility

### WCAG Compliance

```javascript
// Accessibility-focused dark mode implementation
class AccessibleThemeManager {
  constructor() {
    this.contrastRatios = {
      AA: { normal: 4.5, large: 3 },
      AAA: { normal: 7, large: 4.5 }
    };
    
    this.init();
  }
  
  init() {
    // Check for reduced motion preference
    this.prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    
    // Check for high contrast preference
    this.prefersHighContrast = window.matchMedia('(prefers-contrast: high)').matches;
    
    // Initialize with accessible defaults
    this.applyAccessibleTheme();
  }
  
  applyAccessibleTheme() {
    const theme = this.getCurrentTheme();
    
    // Apply base theme
    document.documentElement.setAttribute('data-theme', theme);
    
    // Apply accessibility modifiers
    if (this.prefersHighContrast) {
      document.documentElement.setAttribute('data-contrast', 'high');
    }
    
    if (this.prefersReducedMotion) {
      document.documentElement.setAttribute('data-motion', 'reduced');
    }
    
    // Ensure focus indicators are visible
    this.ensureFocusVisibility();
    
    // Validate color contrasts
    this.validateColorContrasts();
  }
  
  ensureFocusVisibility() {
    // Enhanced focus styles for dark mode
    const style = document.createElement('style');
    style.textContent = `
      [data-theme="dark"] :focus {
        outline: 2px solid #90CAF9;
        outline-offset: 2px;
      }
      
      [data-theme="dark"][data-contrast="high"] :focus {
        outline: 3px solid #FFFFFF;
        outline-offset: 3px;
      }
      
      /* Focus visible only for keyboard navigation */
      [data-theme="dark"] :focus:not(:focus-visible) {
        outline: none;
      }
      
      [data-theme="dark"] :focus-visible {
        outline: 2px solid #90CAF9;
        outline-offset: 2px;
        box-shadow: 0 0 0 4px rgba(144, 202, 249, 0.3);
      }
    `;
    
    document.head.appendChild(style);
  }
  
  validateColorContrasts() {
    // Get all text elements
    const textElements = document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, span, a, button');
    
    textElements.forEach(element => {
      const style = window.getComputedStyle(element);
      const backgroundColor = this.getBackgroundColor(element);
      const textColor = style.color;
      
      const contrast = this.calculateContrast(
        this.parseColor(textColor),
        this.parseColor(backgroundColor)
      );
      
      const fontSize = parseFloat(style.fontSize);
      const isLargeText = fontSize >= 18 || (fontSize >= 14 && style.fontWeight >= 700);
      
      const requiredContrast = isLargeText ? 
        this.contrastRatios.AA.large : 
        this.contrastRatios.AA.normal;
      
      if (contrast < requiredContrast) {
        console.warn(`Low contrast detected: ${contrast.toFixed(2)} (required: ${requiredContrast})`, element);
        
        // Auto-fix if enabled
        if (this.autoFixContrast) {
          this.fixContrast(element, backgroundColor, textColor, requiredContrast);
        }
      }
    });
  }
  
  getBackgroundColor(element) {
    let bg = window.getComputedStyle(element).backgroundColor;
    let parent = element.parentElement;
    
    while (bg === 'transparent' || bg === 'rgba(0, 0, 0, 0)' && parent) {
      bg = window.getComputedStyle(parent).backgroundColor;
      parent = parent.parentElement;
    }
    
    return bg || 'rgb(255, 255, 255)';
  }
  
  calculateContrast(color1, color2) {
    const l1 = this.relativeLuminance(color1);
    const l2 = this.relativeLuminance(color2);
    
    const lighter = Math.max(l1, l2);
    const darker = Math.min(l1, l2);
    
    return (lighter + 0.05) / (darker + 0.05);
  }
  
  relativeLuminance(color) {
    const { r, g, b } = color;
    
    const [rs, gs, bs] = [r, g, b].map(c => {
      c = c / 255;
      return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    });
    
    return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
  }
  
  parseColor(color) {
    const match = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
    if (match) {
      return {
        r: parseInt(match[1]),
        g: parseInt(match[2]),
        b: parseInt(match[3])
      };
    }
    
    // Handle hex colors
    if (color.startsWith('#')) {
      const hex = color.slice(1);
      return {
        r: parseInt(hex.substr(0, 2), 16),
        g: parseInt(hex.substr(2, 2), 16),
        b: parseInt(hex.substr(4, 2), 16)
      };
    }
    
    return { r: 0, g: 0, b: 0 };
  }
  
  fixContrast(element, bgColor, textColor, requiredContrast) {
    const bg = this.parseColor(bgColor);
    const text = this.parseColor(textColor);
    
    // Adjust text color to meet contrast requirements
    const adjustedColor = this.adjustColorForContrast(text, bg, requiredContrast);
    
    element.style.color = `rgb(${adjustedColor.r}, ${adjustedColor.g}, ${adjustedColor.b})`;
    element.setAttribute('data-contrast-fixed', 'true');
  }
  
  adjustColorForContrast(foreground, background, targetContrast) {
    let { r, g, b } = foreground;
    const bgLuminance = this.relativeLuminance(background);
    
    // Determine if we need to lighten or darken
    const currentContrast = this.calculateContrast(foreground, background);
    const shouldLighten = bgLuminance < 0.5;
    
    while (this.calculateContrast({ r, g, b }, background) < targetContrast) {
      if (shouldLighten) {
        r = Math.min(255, r + 10);
        g = Math.min(255, g + 10);
        b = Math.min(255, b + 10);
      } else {
        r = Math.max(0, r - 10);
        g = Math.max(0, g - 10);
        b = Math.max(0, b - 10);
      }
    }
    
    return { r, g, b };
  }
  
  // Announce theme changes to screen readers
  announceThemeChange(theme) {
    const announcement = document.createElement('div');
    announcement.setAttribute('role', 'status');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = `Theme changed to ${theme} mode`;
    
    document.body.appendChild(announcement);
    
    setTimeout(() => {
      document.body.removeChild(announcement);
    }, 1000);
  }
}

// Screen reader only styles
const srOnlyStyles = `
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border-width: 0;
  }
`;
```

## Testing Strategies

### Automated Testing

```javascript
// Jest tests for theme functionality
describe('ThemeManager', () => {
  let themeManager;
  
  beforeEach(() => {
    // Reset DOM
    document.documentElement.removeAttribute('data-theme');
    localStorage.clear();
    
    // Mock matchMedia
    window.matchMedia = jest.fn().mockImplementation(query => ({
      matches: query === '(prefers-color-scheme: dark)',
      media: query,
      onchange: null,
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    }));
    
    themeManager = new ThemeManager();
  });
  
  test('initializes with system preference when no saved theme', () => {
    expect(themeManager.currentTheme).toBe('dark');
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
  });
  
  test('respects saved theme over system preference', () => {
    localStorage.setItem('theme', 'light');
    const newThemeManager = new ThemeManager();
    
    expect(newThemeManager.currentTheme).toBe('light');
  });
  
  test('toggles theme correctly', () => {
    themeManager.setTheme('light');
    expect(themeManager.currentTheme).toBe('light');
    
    themeManager.toggle();
    expect(themeManager.currentTheme).toBe('dark');
    
    themeManager.toggle();
    expect(themeManager.currentTheme).toBe('light');
  });
  
  test('emits theme change events', (done) => {
    window.addEventListener('themechange', (event) => {
      expect(event.detail.theme).toBe('light');
      done();
    });
    
    themeManager.setTheme('light');
  });
  
  test('updates meta theme-color', () => {
    const meta = document.createElement('meta');
    meta.name = 'theme-color';
    document.head.appendChild(meta);
    
    themeManager.setTheme('dark');
    expect(meta.content).toBe('#121212');
    
    themeManager.setTheme('light');
    expect(meta.content).toBe('#FFFFFF');
  });
});

// Cypress E2E tests
describe('Dark Mode E2E', () => {
  beforeEach(() => {
    cy.visit('/');
  });
  
  it('toggles theme when button clicked', () => {
    // Check initial state
    cy.get('html').should('have.attr', 'data-theme', 'light');
    
    // Click toggle
    cy.get('[data-testid="theme-toggle"]').click();
    
    // Check theme changed
    cy.get('html').should('have.attr', 'data-theme', 'dark');
    
    // Check styles applied
    cy.get('body').should('have.css', 'background-color', 'rgb(18, 18, 18)');
  });
  
  it('persists theme preference', () => {
    // Set dark theme
    cy.get('[data-testid="theme-toggle"]').click();
    
    // Reload page
    cy.reload();
    
    // Check theme persisted
    cy.get('html').should('have.attr', 'data-theme', 'dark');
  });
  
  it('respects system preference', () => {
    // Mock dark mode preference
    cy.wrap(window.matchMedia('(prefers-color-scheme: dark)')).then(mq => {
      cy.stub(mq, 'matches').value(true);
    });
    
    cy.visit('/');
    cy.get('html').should('have.attr', 'data-theme', 'dark');
  });
  
  it('transitions smoothly between themes', () => {
    // Enable transitions
    cy.get('body').should('have.css', 'transition');
    
    // Measure transition
    cy.get('[data-testid="theme-toggle"]').click();
    
    // Check no layout shift
    cy.get('.content').should('not.have.class', 'shifting');
  });
});

// Visual regression testing with Percy
describe('Visual Regression', () => {
  it('captures light theme', () => {
    cy.visit('/');
    cy.percySnapshot('Light Theme');
  });
  
  it('captures dark theme', () => {
    cy.visit('/');
    cy.get('[data-testid="theme-toggle"]').click();
    cy.wait(500); // Wait for transitions
    cy.percySnapshot('Dark Theme');
  });
  
  it('captures high contrast mode', () => {
    cy.visit('/?contrast=high');
    cy.percySnapshot('High Contrast Light');
    
    cy.get('[data-testid="theme-toggle"]').click();
    cy.wait(500);
    cy.percySnapshot('High Contrast Dark');
  });
});
```

### Manual Testing Checklist

```markdown
# Dark Mode Testing Checklist

## Functionality
- [ ] Theme toggles correctly
- [ ] Theme persists on page reload
- [ ] System preference detected correctly
- [ ] Theme applies to all components
- [ ] No flash of wrong theme on load

## Visual
- [ ] Colors have sufficient contrast (WCAG AA)
- [ ] Images adapt appropriately
- [ ] Icons remain visible
- [ ] Shadows visible but subtle
- [ ] No harsh transitions
- [ ] Consistent color scheme

## Performance
- [ ] Smooth transitions (60 FPS)
- [ ] No layout shifts
- [ ] Fast theme switching (<100ms)
- [ ] No memory leaks
- [ ] Efficient repaints

## Accessibility
- [ ] Focus indicators visible
- [ ] Screen reader announces changes
- [ ] Keyboard navigation works
- [ ] High contrast mode supported
- [ ] Reduced motion respected

## Cross-browser
- [ ] Chrome/Edge
- [ ] Firefox
- [ ] Safari
- [ ] Mobile browsers
- [ ] IE 11 fallback (if required)

## Edge Cases
- [ ] Multiple theme switches
- [ ] Theme switch during animation
- [ ] Large images loading
- [ ] Offline functionality
- [ ] JavaScript disabled fallback
```

## Real-world Examples

### E-commerce Implementation

```javascript
// E-commerce specific dark mode considerations
class EcommerceDarkMode extends ThemeManager {
  constructor() {
    super();
    this.productImageProcessor = new ProductImageProcessor();
  }
  
  applyTheme(theme) {
    super.applyTheme(theme);
    
    // Handle product images specially
    this.processProductImages(theme);
    
    // Adjust color swatches
    this.adjustColorSwatches(theme);
    
    // Update charts and graphs
    this.updateDataVisualizations(theme);
  }
  
  processProductImages(theme) {
    const productImages = document.querySelectorAll('[data-product-image]');
    
    productImages.forEach(img => {
      if (theme === 'dark') {
        // Add subtle backdrop for better visibility
        img.parentElement.style.backgroundColor = 'rgba(255, 255, 255, 0.05)';
        img.parentElement.style.borderRadius = '8px';
        img.parentElement.style.padding = '8px';
      } else {
        img.parentElement.style.backgroundColor = '';
        img.parentElement.style.padding = '';
      }
    });
  }
  
  adjustColorSwatches(theme) {
    const swatches = document.querySelectorAll('[data-color-swatch]');
    
    swatches.forEach(swatch => {
      const color = swatch.dataset.color;
      
      if (theme === 'dark') {
        // Add border for better visibility of light colors
        if (this.isLightColor(color)) {
          swatch.style.border = '2px solid rgba(255, 255, 255, 0.2)';
        }
      } else {
        // Add border for dark colors in light mode
        if (this.isDarkColor(color)) {
          swatch.style.border = '2px solid rgba(0, 0, 0, 0.1)';
        }
      }
    });
  }
  
  updateDataVisualizations(theme) {
    // Update Chart.js instances
    const charts = window.chartInstances || [];
    
    charts.forEach(chart => {
      chart.options.plugins.legend.labels.color = 
        theme === 'dark' ? 'rgba(255, 255, 255, 0.87)' : 'rgba(0, 0, 0, 0.87)';
      
      chart.options.scales.x.ticks.color = 
        theme === 'dark' ? 'rgba(255, 255, 255, 0.60)' : 'rgba(0, 0, 0, 0.60)';
      
      chart.options.scales.y.ticks.color = 
        theme === 'dark' ? 'rgba(255, 255, 255, 0.60)' : 'rgba(0, 0, 0, 0.60)';
      
      chart.options.scales.x.grid.color = 
        theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
      
      chart.options.scales.y.grid.color = 
        theme === 'dark' ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
      
      chart.update();
    });
  }
  
  isLightColor(color) {
    const rgb = this.hexToRgb(color);
    const brightness = (rgb.r * 299 + rgb.g * 587 + rgb.b * 114) / 1000;
    return brightness > 200;
  }
  
  isDarkColor(color) {
    return !this.isLightColor(color);
  }
  
  hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16)
    } : { r: 0, g: 0, b: 0 };
  }
}

// Product image processor for better dark mode display
class ProductImageProcessor {
  processForDarkMode(img) {
    // Create canvas for processing
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    
    // Draw original image
    ctx.drawImage(img, 0, 0);
    
    // Detect if image has transparent background
    const hasTransparency = this.detectTransparency(ctx, canvas.width, canvas.height);
    
    if (hasTransparency) {
      // Add subtle white background for dark mode
      const processedCanvas = document.createElement('canvas');
      const processedCtx = processedCanvas.getContext('2d');
      
      processedCanvas.width = canvas.width;
      processedCanvas.height = canvas.height;
      
      // Draw white background with slight transparency
      processedCtx.fillStyle = 'rgba(255, 255, 255, 0.03)';
      processedCtx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Draw original image on top
      processedCtx.drawImage(img, 0, 0);
      
      // Replace image source
      img.src = processedCanvas.toDataURL();
    }
  }
  
  detectTransparency(ctx, width, height) {
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;
    
    // Check corners and edges for transparency
    const positions = [
      0, // Top-left
      (width - 1) * 4, // Top-right
      (height - 1) * width * 4, // Bottom-left
      ((height - 1) * width + width - 1) * 4 // Bottom-right
    ];
    
    for (const pos of positions) {
      if (data[pos + 3] < 255) { // Alpha channel
        return true;
      }
    }
    
    return false;
  }
}
```

## Best Practices

### Do's and Don'ts

```javascript
// ✅ DO: Use CSS custom properties
:root {
  --color-primary: #1976D2;
  --color-background: #FFFFFF;
}

[data-theme="dark"] {
  --color-primary: #90CAF9;
  --color-background: #121212;
}

// ✅ DO: Provide smooth transitions
* {
  transition: background-color 0.3s ease, color 0.3s ease;
}

// ✅ DO: Respect user preferences
const theme = localStorage.getItem('theme') || 
             (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');

// ✅ DO: Test color contrasts
function meetsWCAGContrast(foreground, background, level = 'AA') {
  const contrast = calculateContrast(foreground, background);
  return contrast >= (level === 'AAA' ? 7 : 4.5);
}

// ❌ DON'T: Use pure black/white
// Instead of #000000 and #FFFFFF
const colors = {
  dark: '#121212',  // Softer than pure black
  light: '#FAFAFA'  // Softer than pure white
};

// ❌ DON'T: Forget about images
// Always handle images in dark mode
img {
  opacity: 0.9;
  filter: brightness(0.8) contrast(1.2);
}

// ❌ DON'T: Flash incorrect theme
// Prevent FOUC (Flash of Unstyled Content)
<script>
  // Run before page render
  (function() {
    const theme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', theme);
  })();
</script>

// ❌ DON'T: Ignore performance
// Avoid expensive operations during theme switch
function switchTheme(theme) {
  // Bad: Force layout/reflow
  elements.forEach(el => {
    el.style.backgroundColor = getComputedStyle(el).backgroundColor;
  });
  
  // Good: Use CSS variables
  document.documentElement.setAttribute('data-theme', theme);
}
```

### Theme Design Guidelines

```css
/* Professional dark mode color palette */
:root {
  /* Elevation levels for dark mode */
  --dark-elevation-0: #121212;
  --dark-elevation-1: #1E1E1E;
  --dark-elevation-2: #222222;
  --dark-elevation-3: #242424;
  --dark-elevation-4: #272727;
  --dark-elevation-6: #2C2C2C;
  --dark-elevation-8: #2E2E2E;
  --dark-elevation-12: #333333;
  --dark-elevation-16: #363636;
  --dark-elevation-24: #383838;
  
  /* Semantic colors */
  --dark-primary: #BB86FC;
  --dark-primary-variant: #3700B3;
  --dark-secondary: #03DAC6;
  --dark-secondary-variant: #03DAC6;
  --dark-error: #CF6679;
  --dark-warning: #FFB74D;
  --dark-info: #81D4FA;
  --dark-success: #81C784;
  
  /* Text opacity levels */
  --dark-text-high-emphasis: rgba(255, 255, 255, 0.87);
  --dark-text-medium-emphasis: rgba(255, 255, 255, 0.60);
  --dark-text-disabled: rgba(255, 255, 255, 0.38);
  
  /* State overlays */
  --dark-state-hover: rgba(255, 255, 255, 0.04);
  --dark-state-focus: rgba(255, 255, 255, 0.12);
  --dark-state-selected: rgba(255, 255, 255, 0.08);
  --dark-state-activated: rgba(255, 255, 255, 0.12);
  --dark-state-pressed: rgba(255, 255, 255, 0.16);
  --dark-state-dragged: rgba(255, 255, 255, 0.08);
}

/* Component elevation examples */
.card {
  background-color: var(--dark-elevation-1);
}

.card-raised {
  background-color: var(--dark-elevation-4);
}

.dialog {
  background-color: var(--dark-elevation-24);
}

/* Accessible focus states */
[data-theme="dark"] :focus-visible {
  outline: 2px solid var(--dark-primary);
  outline-offset: 2px;
}

/* Handle different content types */
[data-theme="dark"] {
  /* Code blocks */
  pre, code {
    background-color: var(--dark-elevation-2);
    color: var(--dark-text-high-emphasis);
  }
  
  /* Tables */
  table {
    border-color: var(--dark-elevation-4);
  }
  
  th {
    background-color: var(--dark-elevation-2);
  }
  
  tr:nth-child(even) {
    background-color: var(--dark-elevation-1);
  }
  
  /* Forms */
  input, textarea, select {
    background-color: var(--dark-elevation-1);
    border-color: var(--dark-elevation-6);
    color: var(--dark-text-high-emphasis);
  }
  
  input:focus, textarea:focus, select:focus {
    border-color: var(--dark-primary);
    background-color: var(--dark-elevation-2);
  }
  
  /* Scrollbars */
  ::-webkit-scrollbar {
    background-color: var(--dark-elevation-0);
  }
  
  ::-webkit-scrollbar-thumb {
    background-color: var(--dark-elevation-8);
  }
  
  ::-webkit-scrollbar-thumb:hover {
    background-color: var(--dark-elevation-12);
  }
}
```

*Originally from umitkacar/Dark-Mode repository*