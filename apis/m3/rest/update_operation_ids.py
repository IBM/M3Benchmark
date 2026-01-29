#!/usr/bin/env python3
"""
Script to update all FastAPI server files to add operation_id matching the function name.
Also generates metadata files with old->new operation ID mappings.
"""
import os
import re
import json
from pathlib import Path

SERVER_DIR = Path("/Users/anu/Documents/GitHub/routing/EnterpriseBenchmark/apis/m3/rest/server")
METADATA_DIR = SERVER_DIR / "metadata"

# Create metadata directory
METADATA_DIR.mkdir(exist_ok=True)

def extract_function_name(line_after_decorator):
    """Extract function name from the line after the decorator."""
    match = re.search(r'(?:async\s+)?def\s+(\w+)\s*\(', line_after_decorator)
    if match:
        return match.group(1)
    return None

def generate_old_operation_id(path, method, func_name):
    """Generate what FastAPI would auto-generate as operationId."""
    # FastAPI generates: {func_name}_{path_with_underscores}_{method}
    # e.g., get_sum_households_v1_address_sum_households_by_county_get
    path_part = path.replace("/", "_").replace("{", "_").replace("}", "_").strip("_")
    # Remove duplicate underscores
    path_part = re.sub(r'_+', '_', path_part)
    return f"{func_name}_{path_part}_{method}"

def process_file(filepath):
    """Process a single Python file to add operation_ids."""
    with open(filepath, 'r') as f:
        content = f.read()
        lines = content.split('\n')

    mappings = {}
    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this line is a route decorator
        route_match = re.match(r'^(\s*)@app\.(get|post|put|delete|patch)\s*\(\s*["\']([^"\']+)["\'](.*)$', line, re.IGNORECASE)

        if route_match:
            indent = route_match.group(1)
            method = route_match.group(2).lower()
            path = route_match.group(3)
            rest_of_decorator = route_match.group(4)

            # Find the function definition (might be next line or after multi-line decorator)
            func_line_idx = i + 1
            decorator_lines = [line]

            # Handle multi-line decorators
            current_line = line
            open_parens = current_line.count('(') - current_line.count(')')

            while open_parens > 0 and func_line_idx < len(lines):
                decorator_lines.append(lines[func_line_idx])
                current_line = lines[func_line_idx]
                open_parens += current_line.count('(') - current_line.count(')')
                func_line_idx += 1

            # Now func_line_idx should point to the function definition
            if func_line_idx < len(lines):
                func_name = extract_function_name(lines[func_line_idx])

                if func_name:
                    # Generate old operation ID
                    old_op_id = generate_old_operation_id(path, method, func_name)
                    new_op_id = func_name

                    # Store mapping
                    mappings[old_op_id] = new_op_id

                    # Check if operation_id already exists
                    full_decorator = '\n'.join(decorator_lines)
                    if 'operation_id=' not in full_decorator:
                        # Add operation_id to the decorator
                        if len(decorator_lines) == 1:
                            # Single line decorator
                            # Find where to insert operation_id (before the closing paren)
                            if rest_of_decorator.strip() == ')':
                                # Just path, no other params: @app.get("/path")
                                new_line = f'{indent}@app.{method}("{path}", operation_id="{func_name}")'
                            elif rest_of_decorator.strip().startswith(','):
                                # Has other params: @app.get("/path", summary="...")
                                new_line = f'{indent}@app.{method}("{path}", operation_id="{func_name}"{rest_of_decorator}'
                            else:
                                # Unexpected format, try to insert before closing paren
                                if line.rstrip().endswith(')'):
                                    new_line = line.rstrip()[:-1] + f', operation_id="{func_name}")'
                                else:
                                    new_line = line
                            new_lines.append(new_line)
                            i += 1
                            continue
                        else:
                            # Multi-line decorator - add operation_id after the path
                            first_line = decorator_lines[0]
                            # Insert operation_id on first line after path
                            if first_line.rstrip().endswith(','):
                                new_first = first_line.rstrip()[:-1] + f', operation_id="{func_name}",'
                            else:
                                new_first = first_line.rstrip() + f', operation_id="{func_name}",'
                            new_lines.append(new_first)
                            # Add rest of decorator lines
                            for dl in decorator_lines[1:]:
                                new_lines.append(dl)
                            i = func_line_idx
                            continue

        new_lines.append(line)
        i += 1

    return '\n'.join(new_lines), mappings

def main():
    all_files_processed = 0
    all_mappings = {}

    for filepath in sorted(SERVER_DIR.glob("*.py")):
        if filepath.name == "__init__.py":
            continue

        print(f"Processing {filepath.name}...")

        try:
            new_content, mappings = process_file(filepath)

            if mappings:
                # Write updated file
                with open(filepath, 'w') as f:
                    f.write(new_content)

                # Write metadata file
                metadata_file = METADATA_DIR / f"{filepath.stem}_operation_id_map.json"
                with open(metadata_file, 'w') as f:
                    json.dump({
                        "file": filepath.name,
                        "mappings": mappings,
                        "count": len(mappings)
                    }, f, indent=2)

                print(f"  Updated {len(mappings)} routes")
                all_mappings[filepath.name] = mappings
                all_files_processed += 1
            else:
                print(f"  No routes found or already updated")

        except Exception as e:
            print(f"  Error: {e}")

    # Write combined metadata file
    combined_file = METADATA_DIR / "all_operation_id_mappings.json"
    with open(combined_file, 'w') as f:
        json.dump({
            "total_files": all_files_processed,
            "total_mappings": sum(len(m) for m in all_mappings.values()),
            "files": all_mappings
        }, f, indent=2)

    print(f"\nDone! Processed {all_files_processed} files")
    print(f"Total mappings: {sum(len(m) for m in all_mappings.values())}")
    print(f"Metadata saved to: {METADATA_DIR}")

if __name__ == "__main__":
    main()
