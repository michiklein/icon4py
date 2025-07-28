#!/bin/bash

set -eo pipefail

declare -a FILES=()
declare -A RESULTS=()
declare -A PROGRAM_DATA=()

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_color() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

extract_program_info() {
    local filepath="$1"
    local basename=$(basename "$filepath")
    local filename="${basename%.*}"
    
    if [[ $filename =~ ([a-zA-Z0-9_]+)_(on|off)(_32)? ]]; then
        local program="${BASH_REMATCH[1]}"
        local status="${BASH_REMATCH[2]}"
        local block32="${BASH_REMATCH[3]:-}"
        if [[ -n "$block32" ]]; then
            status="${status}_32"
        fi
        echo "$program" "$status"
        return
    fi
    
    if [[ $filename =~ slurm ]]; then
        local program_name=""
        local status=""
        
        while IFS= read -r line; do
            if [[ $line =~ \[[0-9]/8\][[:space:]]*\[[0-9%=]*\][[:space:]]*(v2[ce]2[cev])_(on|off)(_32)?\.nsys-rep ]]; then
                program_name="${BASH_REMATCH[1]}"
                status="${BASH_REMATCH[2]}"
                local block32="${BASH_REMATCH[3]:-}"
                if [[ -n "$block32" ]]; then
                    status="${status}_32"
                fi
                break
            fi
            if [[ $line =~ Generated:[[:space:]]*.*/(v2[ce]2[cev])_(on|off)(_32)?\.nsys-rep ]]; then
                program_name="${BASH_REMATCH[1]}"
                status="${BASH_REMATCH[2]}"
                local block32="${BASH_REMATCH[3]:-}"
                if [[ -n "$block32" ]]; then
                    status="${status}_32"
                fi
                break
            fi
            if [[ $line =~ ([a-zA-Z0-9_]+)_(on|off)(_32)?\.nsys-rep ]]; then
                program_name="${BASH_REMATCH[1]}"
                status="${BASH_REMATCH[2]}"
                local block32="${BASH_REMATCH[3]:-}"
                if [[ -n "$block32" ]]; then
                    status="${status}_32"
                fi
                break
            fi
        done < "$filepath"
        
        if [[ -n "$program_name" && -n "$status" ]]; then
            echo "$program_name" "$status"
            return
        fi
    fi
    
    if [[ $filename =~ ([^_]+)_(on|off)(_32)? ]]; then
        local program="${BASH_REMATCH[1]}"
        local status="${BASH_REMATCH[2]}"
        local block32="${BASH_REMATCH[3]:-}"
        if [[ -n "$block32" ]]; then
            status="${status}_32"
        fi
        echo "$program" "$status"
        return
    fi
    
    echo ""
    print_color $YELLOW "Cannot determine program name and status from filename: $basename"
    read -r -p "Enter program name (e.g., v2c2e, v2e2v): " user_program
    read -r -p "Enter status (on/off/on_32): " user_status
    echo "$user_program" "$user_status"
}

parse_report() {
    local filepath="$1"
    
    if [[ ! -f "$filepath" ]]; then
        print_color $RED "Error: File '$filepath' does not exist."
        return 1
    fi
    
    print_color $BLUE "  Scanning for all programs in file..."
    
    local programs_found=()
    local status=""
    
    while IFS= read -r line; do
        if [[ $line =~ \[[0-9]/8\][[:space:]]*\[[0-9%=]*\][[:space:]]+([a-zA-Z0-9_]+)_(on|off)(_32)?\.nsys-rep ]]; then
            local prog="${BASH_REMATCH[1]}"
            local stat="${BASH_REMATCH[2]}"
            local block32="${BASH_REMATCH[3]:-}"
            if [[ -n "$block32" ]]; then
                stat="${stat}_32"
            fi
            if [[ -z "$status" ]]; then
                status="$stat"
            fi
            if [[ ! " ${programs_found[*]} " =~ " ${prog} " ]]; then
                programs_found+=("$prog")
            fi
        fi
    done < "$filepath"
    
    print_color $YELLOW "  Found programs: ${programs_found[*]} (status: $status)"
    
    local success_count=0
    local current_program=""
    local in_kern_section=false
    local line_number=0
    
    while IFS= read -r line; do
        line_number=$((line_number + 1))
        
        if [[ $line =~ \[[0-9]/8\][[:space:]]*\[[0-9%=]*\][[:space:]]+([a-zA-Z0-9_]+)_(on|off)(_32)?\.nsys-rep ]]; then
            current_program="${BASH_REMATCH[1]}"
            local stat="${BASH_REMATCH[2]}"
            local block32="${BASH_REMATCH[3]:-}"
            if [[ -n "$block32" ]]; then
                stat="${stat}_32"
            fi
            status="$stat"
            in_kern_section=false
            print_color $BLUE "  Starting section for: ${current_program}_${status}"
        fi
        
        if [[ $line =~ ^\[6/8\].*cuda_gpu_kern_sum ]] && [[ -n "$current_program" ]]; then
            in_kern_section=true
            print_color $YELLOW "    Found cuda_gpu_kern_sum section for $current_program"
            continue
        fi
        
        if [[ $in_kern_section == true && $line =~ ^\[[78]/8\] ]]; then
            in_kern_section=false
            continue
        fi
        
        if [[ $in_kern_section == true && $line =~ void.*gridtools && -n "$current_program" ]]; then
            print_color $GREEN "    Found gridtools line for $current_program"
            
            local numbers=($line)
            if [[ ${#numbers[@]} -ge 5 ]]; then
                local total_time_ns="${numbers[1]//,/}"
                local median_time_ns="${numbers[4]//,/}"
                
                local median_time_us=$(echo "scale=2; $median_time_ns / 1000" | bc -l)
                local median_time_ms=$(echo "scale=4; $median_time_ns / 1000000" | bc -l)
                
                local key="${current_program}_${status}"
                RESULTS["${key}_total_ns"]="$total_time_ns"
                RESULTS["${key}_median_ns"]="$median_time_ns"
                RESULTS["${key}_median_us"]="$median_time_us"
                RESULTS["${key}_median_ms"]="$median_time_ms"
                RESULTS["${key}_program"]="$current_program"
                RESULTS["${key}_status"]="$status"
                RESULTS["${key}_filepath"]="$filepath"
                
                PROGRAM_DATA["$current_program"]=1
                
                print_color $GREEN "    ✓ Found ${current_program}_${status} - Median: ${median_time_us} μs"
                success_count=$((success_count + 1))
                in_kern_section=false
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

format_number() {
    local num="$1"
    printf "%'d" "$num" 2>/dev/null || echo "$num"
}

calculate_speedup() {
    local program="$1"
    local on_time="${RESULTS[${program}_on_median_us]:-}"
    local on_32_time="${RESULTS[${program}_on_32_median_us]:-}"
    local off_time="${RESULTS[${program}_off_median_us]:-}"
    
    if [[ -n "$on_time" && -n "$off_time" && $(echo "$on_time > 0" | bc -l) -eq 1 ]]; then
        local speedup=$(echo "scale=2; $off_time / $on_time" | bc -l)
        echo "             on vs off: ${speedup}x"
    fi
    
    if [[ -n "$on_time" && -n "$on_32_time" && $(echo "$on_time > 0" | bc -l) -eq 1 ]]; then
        local speedup=$(echo "scale=2; $on_time / $on_32_time" | bc -l)
        echo "             on_32 vs on: ${speedup}x"
    fi
}

display_results() {
    if [[ ${#RESULTS[@]} -eq 0 ]]; then
        echo "No valid results to display."
        return
    fi
    
    print_color $BLUE "GPU Kernel Performance Comparison (void gridtools)"
    echo "================================================================================"
    echo ""
    
    printf "%-12s %-10s %-16s\n" "Program" "Status" "Median Time (μs)"
    echo "--------------------------------------------------------------------------------"
    
    for program in $(printf '%s\n' "${!PROGRAM_DATA[@]}" | sort); do
        for status in on on_32 off; do
            local key="${program}_${status}"
            if [[ -n "${RESULTS[${key}_median_us]:-}" ]]; then
                printf "%-12s %-10s %-16s\n" \
                    "$program" \
                    "$status" \
                    "${RESULTS[${key}_median_us]}"
            fi
        done
        
        calculate_speedup "$program"
        echo ""
    done
}

check_dependencies() {
    if ! command -v bc &> /dev/null; then
        print_color $RED "Error: 'bc' calculator is required but not installed."
        echo "Please install bc: sudo apt-get install bc (Ubuntu/Debian) or brew install bc (macOS)"
        exit 1
    fi
}

main() {
    check_dependencies
    
    if [[ $# -gt 0 ]]; then
        FILES=("$@")
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
    
    if [[ $success_count -gt 0 ]] || [[ ${#RESULTS[@]} -gt 0 ]]; then
        echo ""
        display_results
    else
        print_color $RED "No valid results found. Please check your files contain the expected format."
        exit 1
    fi
}

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
  - Files should follow pattern: {program}_{on|off|on_32}.{extension}
  - Examples: v2c2e_on.nsys-rep, v2e2v_off.txt, v2c2e_on_32.nsys-rep, slurm-12345.out
  - For slurm.out files, the script will try to detect program info from content
  - _32 suffix indicates block-sorted (32-element blocks) configuration

Status types:
  - on: Regular configuration with collapse tables enabled
  - off: Regular configuration with collapse tables disabled  
  - on_32: Block-sorted configuration with collapse tables enabled

EOF
    exit 0
fi

main "$@"