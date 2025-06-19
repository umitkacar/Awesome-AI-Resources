#!/usr/bin/env python3
"""
Auto-update repository statistics in README.md
"""

import os
import re
import json
import requests
from datetime import datetime
from pathlib import Path

def count_resources():
    """Count total resources across all markdown files"""
    total_links = 0
    total_files = 0
    
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.endswith('.md') and file not in ['README.md', 'CONTRIBUTING.md', 'ROADMAP.md']:
                total_files += 1
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Count markdown links [text](url)
                    links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
                    # Filter out internal links
                    external_links = [link for link in links if link[1].startswith('http')]
                    total_links += len(external_links)
    
    return total_files, total_links

def count_categories():
    """Count main categories (top-level directories)"""
    categories = 0
    for item in os.listdir('.'):
        if os.path.isdir(item) and not item.startswith('.') and item not in ['scripts', 'notebooks', 'web']:
            categories += 1
    return categories

def get_github_stats():
    """Get GitHub repository statistics"""
    # This would need GitHub API token in production
    # For now, return placeholder
    return {
        'stars': '1+',
        'forks': '0',
        'contributors': '1'
    }

def update_readme():
    """Update README.md with latest statistics"""
    readme_path = 'README.md'
    
    # Get current stats
    total_files, total_resources = count_resources()
    total_categories = count_categories()
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Read README
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update statistics line
    stats_pattern = r'\*\*Last Updated:\*\* .+ \| \*\*Total Resources:\*\* \d+\+ \| \*\*Categories:\*\* \d+'
    new_stats = f'**Last Updated:** {current_date} | **Total Resources:** {total_resources}+ | **Categories:** {total_categories}'
    
    content = re.sub(stats_pattern, new_stats, content)
    
    # Write back
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Updated README.md:")
    print(f"   - Date: {current_date}")
    print(f"   - Resources: {total_resources}+")
    print(f"   - Categories: {total_categories}")
    print(f"   - Files: {total_files}")

def create_stats_badge():
    """Create statistics badge JSON for shields.io"""
    total_files, total_resources = count_resources()
    
    stats = {
        "schemaVersion": 1,
        "label": "resources",
        "message": f"{total_resources}+",
        "color": "brightgreen"
    }
    
    os.makedirs('.github/badges', exist_ok=True)
    with open('.github/badges/resources.json', 'w') as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    print("📊 Updating repository statistics...")
    update_readme()
    create_stats_badge()
    print("✨ Statistics updated successfully!")