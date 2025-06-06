#!/bin/bash

# NVIDIA Nsight Systems Report Comparison Tool (Bash Version)
# 
# This script parses multiple NSYS profiling reports and compares GPU kernel performance,
# specifically focusing on the 'void gridtools' kernel execution times across different
# test configurations.

set -eo pipefail

# Global variables
declare -a FILES=()
declare -A RESULTS=()
declare -A PROGRAM_DATA=()

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

# Function to extract program name and status from filename or file content
extract_program_info() {
    local filepath="$1"
    local basename=$(basename "$filepath")
    local filename="${basename%.*}"
    
    # First try to extract from filename patterns
    # Try to match pattern like 'program_on' or 'program_off' for any program name
    if [[ $filename =~ ([a-zA-Z0-9_]+)_(on|off) ]]; then
        echo "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
        return
    fi
    
    # Try pattern like 'slurm-12345' and look inside file for program info
    if [[ $filename =~ slurm ]]; then
        # Look for program info in the file content
        local program_name=""
        local status=""
        
        # Look for pattern in progress lines like [1/8] [0%] v2c2e_on.nsys-rep
        while IFS= read -r line; do
            # Match progress lines with .nsys-rep filenames
            if [[ $line =~ \[[0-9]/8\][[:space:]]*\[[0-9%=]*\][[:space:]]*(v2[ce]2[cev])_(on|off)\.nsys-rep ]]; then
                program_name="${BASH_REMATCH[1]}"
                status="${BASH_REMATCH[2]}"
                break
            fi
            # Also look for Generated files that might indicate the program
            if [[ $line =~ Generated:[[:space:]]*.*/(v2[ce]2[cev])_(on|off)\.nsys-rep ]]; then
                program_name="${BASH_REMATCH[1]}"
                status="${BASH_REMATCH[2]}"
                break
            fi
            # Alternative: look for any mention of pattern in the output
            if [[ $line =~ ([a-zA-Z0-9_]+)_(on|off)\.nsys-rep ]]; then
                program_name="${BASH_REMATCH[1]}"
                status="${BASH_REMATCH[2]}"
                break
            fi
        done < "$filepath"
        
        if [[ -n "$program_name" && -n "$status" ]]; then
            echo "$program_name" "$status"
            return
        fi
    fi
    
    # Fallback: try to match any pattern before _on/_off
    if [[ $filename =~ ([^_]+)_(on|off) ]]; then
        echo "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
        return
    fi
    
    # If no clear pattern, prompt user for clarification
    echo ""
    print_color $YELLOW "Cannot determine program name and status from filename: $basename"
    read -r -p "Enter program name (e.g., v2c2e, v2e2v): " user_program
    read -r -p "Enter status (on/off): " user_status
    echo "$user_program" "$user_status"
}

# Function to parse a single NSYS report - now handles multiple programs per file
parse_report() {
    local filepath="$1"
    
    if [[ ! -f "$filepath" ]]; then
        print_color $RED "Error: File '$filepath' does not exist."
        return 1
    fi
    
    print_color $BLUE "  Scanning for all programs in file..."
    
    # Find all programs mentioned in the file
    local programs_found=()
    local status=""
    
    while IFS= read -r line; do
        if [[ $line =~ \[[0-9]/8\][[:space:]]*\[[0-9%=]*\][[:space:]]+([a-zA-Z0-9_]+)_(on|off)\.nsys-rep ]]; then
            local prog="${BASH_REMATCH[1]}"
            local stat="${BASH_REMATCH[2]}"
            if [[ -z "$status" ]]; then
                status="$stat"  # Set status from first match
            fi
            # Add to programs list if not already there
            if [[ ! " ${programs_found[*]} " =~ " ${prog} " ]]; then
                programs_found+=("$prog")
            fi
        fi
    done < "$filepath"
    
    print_color $YELLOW "  Found programs: ${programs_found[*]} (status: $status)"
    
    # Now find each program's cuda_gpu_kern_sum section
    local success_count=0
    local current_program=""
    local in_kern_section=false
    local line_number=0
    
    while IFS= read -r line; do
        line_number=$((line_number + 1))
        
        # Check if we're starting a new program section
        if [[ $line =~ \[[0-9]/8\][[:space:]]*\[[0-9%=]*\][[:space:]]+([a-zA-Z0-9_]+)_(on|off)\.nsys-rep ]]; then
            current_program="${BASH_REMATCH[1]}"
            in_kern_section=false
            print_color $BLUE "  Starting section for: ${current_program}_${status}"
        fi
        
        # Check if we're entering a cuda_gpu_kern_sum section
        if [[ $line =~ ^\[6/8\].*cuda_gpu_kern_sum ]] && [[ -n "$current_program" ]]; then
            in_kern_section=true
            print_color $YELLOW "    Found cuda_gpu_kern_sum section for $current_program"
            continue
        fi
        
        # Check if we're leaving the section
        if [[ $in_kern_section == true && $line =~ ^\[[78]/8\] ]]; then
            in_kern_section=false
            continue
        fi
        
        # If we're in the section, look for gridtools kernel
        if [[ $in_kern_section == true && $line =~ void.*gridtools && -n "$current_program" ]]; then
            print_color $GREEN "    Found gridtools line for $current_program"
            
            # Extract numbers using array splitting
            local numbers=($line)
            if [[ ${#numbers[@]} -ge 5 ]]; then
                local total_time_ns="${numbers[1]//,/}"  # Second column (Total Time)
                local median_time_ns="${numbers[4]//,/}" # Fifth column (Med)
                
                # Calculate microseconds and milliseconds
                local median_time_us=$(echo "scale=2; $median_time_ns / 1000" | bc -l)
                local median_time_ms=$(echo "scale=4; $median_time_ns / 1000000" | bc -l)
                
                # Store results
                local key="${current_program}_${status}"
                RESULTS["${key}_total_ns"]="$total_time_ns"
                RESULTS["${key}_median_ns"]="$median_time_ns"
                RESULTS["${key}_median_us"]="$median_time_us"
                RESULTS["${key}_median_ms"]="$median_time_ms"
                RESULTS["${key}_program"]="$current_program"
                RESULTS["${key}_status"]="$status"
                RESULTS["${key}_filepath"]="$filepath"
                
                # Track programs for later processing
                PROGRAM_DATA["$current_program"]=1
                
                print_color $GREEN "    ✓ Found ${current_program}_${status} - Median: ${median_time_us} μs"
                success_count=$((success_count + 1))
                in_kern_section=false  # Stop looking in this section
            fi
        fi
    done < "$filepath"
    
    if [[ $success_count -gt 0 ]]; then
        print_color $GREEN "  Successfully extracted $success_count programs from file"
        return 0
    else
        print_color $RED "  No gridtools kernels found in any sections"
        return 1
    fi
}

# Function to prompt for files interactively
prompt_for_files() {
    print_color $BLUE "NSYS Report Comparison Tool"
    echo "========================================"
    echo "Enter the paths to your NSYS report files (slurm.out files)."
    echo "You can enter multiple files, one per line."
    echo "Press Enter on empty line when done, or type 'quit' to exit."
    echo ""
    
    local file_count=0
    while true; do
        read -r -p "File $((file_count + 1)): " file_path
        
        if [[ "$file_path" == "quit" ]]; then
            exit 0
        fi
        
        if [[ -z "$file_path" ]]; then
            if [[ $file_count -ge 2 ]]; then
                break
            else
                echo "Please enter at least 2 files for comparison."
                continue
            fi
        fi
        
        if [[ ! -f "$file_path" ]]; then
            print_color $YELLOW "Warning: File '$file_path' does not exist. Please check the path."
            continue
        fi
        
        FILES+=("$file_path")
        echo "Added: $file_path"
        file_count=$((file_count + 1))
    done
}

# Function to format numbers with commas
format_number() {
    local num="$1"
    printf "%'d" "$num" 2>/dev/null || echo "$num"
}

# Function to calculate speedup and format comparison
calculate_speedup() {
    local program="$1"
    local on_time="${RESULTS[${program}_on_median_us]:-}"
    local off_time="${RESULTS[${program}_off_median_us]:-}"
    
    if [[ -n "$on_time" && -n "$off_time" && $(echo "$on_time > 0" | bc -l) -eq 1 ]]; then
        local speedup=$(echo "scale=2; $off_time / $on_time" | bc -l)
        echo "             Speedup: ${speedup}x"
    fi
}

# Function to display comparison table
display_results() {
    if [[ ${#RESULTS[@]} -eq 0 ]]; then
        echo "No valid results to display."
        return
    fi
    
    print_color $BLUE "GPU Kernel Performance Comparison (void gridtools)"
    echo "================================================================================"
    echo ""
    
    # Header
    printf "%-12s %-8s %-16s\n" "Program" "Status" "Median Time (μs)"
    echo "--------------------------------------------------------------------------------"
    
    # Process each program
    for program in $(printf '%s\n' "${!PROGRAM_DATA[@]}" | sort); do
        # Show 'on' first, then 'off'
        for status in on off; do
            local key="${program}_${status}"
            if [[ -n "${RESULTS[${key}_median_us]:-}" ]]; then
                printf "%-12s %-8s %-16s\n" \
                    "$program" \
                    "$status" \
                    "${RESULTS[${key}_median_us]}"
            fi
        done
        
        # Add speedup calculation if both on/off exist
        calculate_speedup "$program"
        echo ""
    done
}

# Function to check dependencies
check_dependencies() {
    if ! command -v bc &> /dev/null; then
        print_color $RED "Error: 'bc' calculator is required but not installed."
        echo "Please install bc: sudo apt-get install bc (Ubuntu/Debian) or brew install bc (macOS)"
        exit 1
    fi
}

# Main function
main() {
    check_dependencies
    
    # Parse command line arguments or prompt for files
    if [[ $# -gt 0 ]]; then
        FILES=("$@")
        # Validate files exist
        for file_path in "${FILES[@]}"; do
            if [[ ! -f "$file_path" ]]; then
                print_color $RED "Error: File '$file_path' does not exist."
                exit 1
            fi
        done
    else
        prompt_for_files
    fi
    
    if [[ ${#FILES[@]} -lt 2 ]]; then
        print_color $RED "Error: At least 2 files are required for comparison."
        exit 1
    fi
    
    # Parse reports
    echo ""
    print_color $BLUE "Parsing ${#FILES[@]} files..."
    local success_count=0
    
    for file_path in "${FILES[@]}"; do
        echo "Processing: $file_path"
        if parse_report "$file_path"; then
            success_count=$((success_count + 1))
        else
            print_color $RED "  ✗ Failed to parse or no gridtools kernel found"
        fi
    done
    
    echo ""
    print_color $GREEN "Successfully parsed $success_count files."
    echo ""
    
    # Display results - check both success_count and actual results
    if [[ $success_count -gt 0 ]] || [[ ${#RESULTS[@]} -gt 0 ]]; then
        echo ""
        display_results
    else
        print_color $RED "No valid results found. Please check your files contain the expected format."
        exit 1
    fi
}

# Show usage if --help is passed
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat << EOF
NSYS Report Comparison Tool

Usage: $0 [file1] [file2] [file3] ...

If no files are provided, the script will prompt interactively for file paths.

Examples:
  $0                                    # Interactive mode
  $0 v2c2e_on.txt v2c2e_off.txt       # Compare two files
  $0 *.txt                             # Compare all .txt files

Requirements:
  - bc calculator (for floating point arithmetic)

The script expects files to contain NSYS profiling output with:
  - A '[6/8] Executing 'cuda_gpu_kern_sum' stats report' section
  - A line containing 'void gridtools' with performance data

File naming convention:
  - Files should follow pattern: {program}_{on|off}.{extension}
  - Examples: v2c2e_on.nsys-rep, v2e2v_off.txt, slurm-12345.out
  - For slurm.out files, the script will try to detect program info from content

EOF
    exit 0
fi

# Run main function
main "$@"