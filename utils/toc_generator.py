#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table of Contents (TOC) Generator for Markdown Files

This module provides functionality to generate a table of contents
for markdown files with collapsible sections for headings with children.
"""

import re
import os
import argparse
from typing import List, Tuple, Dict, Any

def extract_headings(markdown_content: str) -> List[Dict[str, Any]]:
    """
    Extract headings from a markdown file.
    
    Args:
        markdown_content (str): The content of the markdown file
        
    Returns:
        List[Dict]: List of dictionaries containing heading information
    """
    # Regular expression to match markdown headings
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    headings = []
    for match in heading_pattern.finditer(markdown_content):
        level = len(match.group(1))
        title = match.group(2).strip()
        headings.append({
            'level': level,
            'title': title,
            'slug': create_slug(title, level)
        })
    
    return headings

def create_slug(title: str, level: int = None) -> str:
    """
    Create a slug from a heading title.
    
    Args:
        title (str): The heading title
        level (int, optional): The heading level
        
    Returns:
        str: The slug
    """
    # Remove any special characters and convert to lowercase
    slug = title.lower()
    
    # Replace spaces with hyphens
    slug = re.sub(r'\s+', '-', slug)
    
    # Remove any non-alphanumeric characters (except hyphens)
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    
    # Remove consecutive hyphens
    slug = re.sub(r'-+', '-', slug)
    
    # Remove leading and trailing hyphens
    slug = slug.strip('-')
    
    return slug

def organize_headings(headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Organize headings into a hierarchical structure.
    
    Args:
        headings (List[Dict]): List of dictionaries containing heading information
        
    Returns:
        List[Dict]: List of dictionaries containing organized heading information
    """
    # Initialize variables
    result = []
    current_h1 = None
    h1_count = 0
    h2_count = 0
    
    for heading in headings:
        if heading['level'] == 1:
            # Create a new H1 heading
            h1_count += 1
            h2_count = 0
            current_h1 = {
                'level': 1,
                'number': h1_count,
                'title': heading['title'],
                'slug': heading['slug'],
                'children': []
            }
            result.append(current_h1)
        elif heading['level'] == 2 and current_h1 is not None:
            # Add an H2 heading to the current H1
            h2_count += 1
            h2_heading = {
                'level': 2,
                'number': f"{current_h1['number']}.{h2_count}",
                'title': heading['title'],
                'slug': heading['slug']
            }
            current_h1['children'].append(h2_heading)
    
    return result

def generate_toc(organized_headings: List[Dict[str, Any]]) -> str:
    """
    Generate a table of contents from organized headings.
    
    Args:
        organized_headings (List[Dict]): List of dictionaries containing organized heading information
        
    Returns:
        str: The table of contents as HTML
    """
    toc = []
    
    for h1_index, h1 in enumerate(organized_headings):
        if not h1['children']:
            # H1 without children - simple non-collapsible format
            toc.append(f'<div>\n  &nbsp;&nbsp;&nbsp;&nbsp;<a href="#{h1["number"]}-{h1["slug"]}"><i><b>{h1["number"]}. {h1["title"]}</b></i></a>\n</div>\n&nbsp;\n')
        else:
            # H1 with children - collapsible format
            toc.append(f'<details>\n  <summary><a href="#{h1["number"]}-{h1["slug"]}"><i><b>{h1["number"]}. {h1["title"]}</b></i></a></summary>\n  <div>')
            
            # Add H2 children
            for h2 in h1['children']:
                toc.append(f'    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="#{h2["number"].replace(".", "")}-{h2["slug"]}">{h2["number"]}. {h2["title"]}</a><br>')
            
            toc.append('  </div>\n</details>\n&nbsp;\n')
    
    return '\n'.join(toc)

def process_markdown_file(file_path: str) -> str:
    """
    Process a markdown file and generate a table of contents.
    
    Args:
        file_path (str): Path to the markdown file
        
    Returns:
        str: The generated table of contents
    """
    # Read the markdown file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Extract headings
    headings = extract_headings(content)
    
    # Organize headings
    organized_headings = organize_headings(headings)
    
    # Generate table of contents
    toc = generate_toc(organized_headings)
    
    return toc

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate a table of contents for a markdown file')
    parser.add_argument('file', help='Path to the markdown file')
    parser.add_argument('--output', '-o', help='Output file path (default: stdout)')
    
    args = parser.parse_args()
    
    toc = process_markdown_file(args.file)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as file:
            file.write(toc)
        print(f"Table of contents written to {args.output}")
    else:
        print(toc)

if __name__ == '__main__':
    main() 