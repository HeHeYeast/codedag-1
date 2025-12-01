#!/bin/bash
# Docker环境中的测试运行脚本

# 设置环境变量
export PYTHONPATH=/codedag_tests
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_msg() {
    echo -e "${GREEN}[DOCKER-INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[DOCKER-ERROR]${NC} $1"
}

print_section() {
    echo -e "${BLUE}[DOCKER-SECTION]${NC} $1"
}

# 系统信息检查
check_environment() {
    print_section "Checking Docker environment..."
    
    echo "Python version: $(python --version)"
    echo "Current directory: $(pwd)"
    echo "PYTHONPATH: $PYTHONPATH"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    
    # 检查GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits | head -4
    else
        echo "nvidia-smi not available"
    fi
    
    # 检查Python包
    print_msg "Checking required packages..."
    python -c "
import sys
packages = ['torch', 'numpy', 'psutil']
for pkg in packages:
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f'✓ {pkg}: {version}')
    except ImportError:
        print(f'✗ {pkg}: NOT FOUND')
        sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_msg "Environment check passed!"
    else
        print_error "Environment check failed!"
        exit 1
    fi
}

# 运行单个测试
run_single_test() {
    local test_file=$1
    local test_name=$2
    
    print_section "Running $test_name..."
    
    if [ ! -f "$test_file" ]; then
        print_error "Test file not found: $test_file"
        return 1
    fi
    
    # 创建结果目录
    mkdir -p test_results
    
    # 运行测试
    echo "Executing: python $test_file"
    python "$test_file"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_msg "$test_name completed successfully!"
        return 0
    else
        print_error "$test_name failed with exit code $exit_code"
        return 1
    fi
}

# 运行所有测试
run_all_tests() {
    print_section "Running all dataflow parsing tests..."
    
    local tests=(
        "examples/dataflow_parsing_tests/test_basic_arithmetic.py:Basic Arithmetic"
        "examples/dataflow_parsing_tests/test_complex_computation.py:Complex Computation"
        "examples/dataflow_parsing_tests/test_numpy_framework.py:NumPy Framework"
        "examples/dataflow_parsing_tests/test_pytorch_framework.py:PyTorch Framework"
    )
    
    local passed=0
    local total=${#tests[@]}
    
    for test_info in "${tests[@]}"; do
        IFS=':' read -r test_file test_name <<< "$test_info"
        
        if run_single_test "$test_file" "$test_name"; then
            ((passed++))
        fi
        
        echo "----------------------------------------"
    done
    
    print_section "Test Summary"
    echo "Passed: $passed/$total tests"
    
    if [ $passed -eq $total ]; then
        print_msg "All tests passed!"
        return 0
    else
        print_error "Some tests failed!"
        return 1
    fi
}

# 显示测试结果
show_results() {
    print_section "Test Results Summary"
    
    if [ ! -d "test_results" ]; then
        print_error "No test results found!"
        return 1
    fi
    
    echo "Result files:"
    ls -la test_results/
    
    echo ""
    echo "JSON Results Summary:"
    for json_file in test_results/*.json; do
        if [ -f "$json_file" ]; then
            echo "--- $(basename $json_file) ---"
            python -c "
import json
try:
    with open('$json_file', 'r') as f:
        data = json.load(f)
    meta = data.get('metadata', {})
    print(f'  Nodes: {meta.get(\"total_nodes\", \"N/A\")}')
    print(f'  Edges: {meta.get(\"total_edges\", \"N/A\")}')
    print(f'  GPU Operations: {meta.get(\"gpu_operations\", \"N/A\")}')
    print(f'  Trace Depth: {meta.get(\"trace_depth\", \"N/A\")}')
    ops = meta.get('traced_operations', [])
    print(f'  Operations: {len(ops)} total')
    if len(ops) > 0:
        print(f'  First 3 ops: {ops[:3]}')
except Exception as e:
    print(f'  Error reading file: {e}')
"
        fi
    done
}

# 交互式模式
interactive_mode() {
    print_section "Interactive Test Mode"
    echo "Available commands:"
    echo "  1. run_test <test_name>"
    echo "  2. show_results"
    echo "  3. check_env"
    echo "  4. exit"
    echo ""
    
    while true; do
        read -p "docker-test> " command args
        
        case $command in
            run_test)
                case $args in
                    basic)
                        run_single_test "examples/dataflow_parsing_tests/test_basic_arithmetic.py" "Basic Arithmetic"
                        ;;
                    complex)
                        run_single_test "examples/dataflow_parsing_tests/test_complex_computation.py" "Complex Computation"
                        ;;
                    numpy)
                        run_single_test "examples/dataflow_parsing_tests/test_numpy_framework.py" "NumPy Framework"
                        ;;
                    pytorch)
                        run_single_test "examples/dataflow_parsing_tests/test_pytorch_framework.py" "PyTorch Framework"
                        ;;
                    all)
                        run_all_tests
                        ;;
                    *)
                        echo "Usage: run_test [basic|complex|numpy|pytorch|all]"
                        ;;
                esac
                ;;
            show_results)
                show_results
                ;;
            check_env)
                check_environment
                ;;
            exit|quit)
                print_msg "Exiting interactive mode"
                break
                ;;
            help|*)
                echo "Available commands: run_test, show_results, check_env, exit"
                ;;
        esac
        echo ""
    done
}

# 主程序
main() {
    echo "CodeDAG Dataflow Tests - Docker Runner"
    echo "====================================="
    
    # 先检查环境
    check_environment
    echo ""
    
    case ${1:-help} in
        check)
            check_environment
            ;;
        basic)
            run_single_test "examples/dataflow_parsing_tests/test_basic_arithmetic.py" "Basic Arithmetic"
            ;;
        complex)
            run_single_test "examples/dataflow_parsing_tests/test_complex_computation.py" "Complex Computation"
            ;;
        numpy)
            run_single_test "examples/dataflow_parsing_tests/test_numpy_framework.py" "NumPy Framework"
            ;;
        pytorch)
            run_single_test "examples/dataflow_parsing_tests/test_pytorch_framework.py" "PyTorch Framework"
            ;;
        all)
            run_all_tests
            ;;
        results)
            show_results
            ;;
        interactive)
            interactive_mode
            ;;
        help|*)
            echo "Usage: $0 [check|basic|complex|numpy|pytorch|all|results|interactive]"
            echo ""
            echo "Commands:"
            echo "  check       - Check environment"
            echo "  basic       - Run basic arithmetic test"
            echo "  complex     - Run complex computation test"
            echo "  numpy       - Run NumPy framework test"
            echo "  pytorch     - Run PyTorch framework test"
            echo "  all         - Run all tests"
            echo "  results     - Show test results"
            echo "  interactive - Interactive mode"
            ;;
    esac
}

main $@