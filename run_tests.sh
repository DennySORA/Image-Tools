#!/bin/bash
# 測試執行腳本

set -e

echo "🧪 Remove-Background 測試套件"
echo "=============================="
echo ""

# 顏色定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 檢查參數
TEST_TYPE="${1:-fast}"

case "$TEST_TYPE" in
    fast)
        echo -e "${GREEN}運行快速測試（跳過慢速測試）${NC}"
        pytest -m "not slow" -v
        ;;
    all)
        echo -e "${YELLOW}運行所有測試（包含慢速測試）${NC}"
        pytest -v
        ;;
    unit)
        echo -e "${GREEN}運行單元測試${NC}"
        pytest -m "unit and not slow" -v
        ;;
    integration)
        echo -e "${GREEN}運行整合測試${NC}"
        pytest -m "integration and not slow" -v
        ;;
    e2e)
        echo -e "${YELLOW}運行端到端測試${NC}"
        pytest -m "e2e" -v
        ;;
    coverage)
        echo -e "${GREEN}運行測試並生成覆蓋率報告${NC}"
        pytest -m "not slow" --cov=src --cov-report=html --cov-report=term
        echo ""
        echo -e "${GREEN}覆蓋率報告已生成：htmlcov/index.html${NC}"
        ;;
    *)
        echo -e "${RED}未知的測試類型: $TEST_TYPE${NC}"
        echo ""
        echo "使用方式："
        echo "  ./run_tests.sh [TYPE]"
        echo ""
        echo "可用類型："
        echo "  fast        - 快速測試（預設，跳過慢速測試）"
        echo "  all         - 所有測試（包含慢速測試）"
        echo "  unit        - 單元測試"
        echo "  integration - 整合測試"
        echo "  e2e         - 端到端測試"
        echo "  coverage    - 測試 + 覆蓋率報告"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✅ 測試完成！${NC}"
