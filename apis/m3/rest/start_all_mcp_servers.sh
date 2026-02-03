#!/bin/bash
#
# Start one MCP server per domain (80 total).
# Each server filters to /v1/{domain}/* endpoints from the shared FastAPI server.
#
# Usage:
#   ./start_all_mcp_servers.sh                    # Start all 80 servers
#   ./start_all_mcp_servers.sh start hockey movie  # Start specific domains only
#   ./start_all_mcp_servers.sh stop               # Stop all servers
#   ./start_all_mcp_servers.sh status             # Check status
#   ./start_all_mcp_servers.sh restart            # Restart all
#   ./start_all_mcp_servers.sh list               # Show all domains + ports
#   ./start_all_mcp_servers.sh logs hockey        # Tail logs for a domain
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FASTAPI_BASE_URL="${FASTAPI_BASE_URL:-http://localhost:8000}"
LOG_DIR="${LOG_DIR:-./logs/mcp}"
PID_DIR="${PID_DIR:-./pids}"
BASE_PORT="${BASE_PORT:-8001}"

# All 80 domains (sorted alphabetically, matching server/*.py modules).
# Port = BASE_PORT + index (address=8001 ... world_development_indicators=8080)
DOMAINS=(
    address
    airline
    app_store
    authors
    beer_factory
    bike_share_1
    book_publishing_company
    books
    california_schools
    car_retails
    card_games
    cars
    chicago_crime
    citeseer
    codebase_comments
    codebase_community
    coinmarketcap
    college_completion
    computer_student
    cookbook
    craftbeer
    cs_semester
    debit_card_specializing
    disney
    donor
    european_football_1
    european_football_2
    financial
    food_inspection
    food_inspection_2
    formula_1
    genes
    hockey
    human_resources
    ice_hockey_draft
    image_and_language
    language_corpus
    law_episode
    legislator
    mental_health_survey
    menu
    mondial_geo
    movie
    movie_3
    movie_platform
    movies_4
    movielens
    music_platform_2
    music_tracker
    olympics
    professional_basketball
    public_review_platform
    regional_sales
    restaurant
    retail_complains
    retail_world
    retails
    sales
    sales_in_weather
    shakespeare
    shipping
    shooting
    simpson_episodes
    soccer_2016
    social_media
    software_company
    student_club
    student_loan
    superhero
    superstore
    synthea
    talkingdata
    thrombosis_prediction
    toxicology
    trains
    university
    video_games
    works_cycles
    world
    world_development_indicators
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC}   $1"; }
log_warning() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

setup_dirs() { mkdir -p "$LOG_DIR" "$PID_DIR"; }

check_fastapi() {
    log_info "Checking FastAPI server at $FASTAPI_BASE_URL..."
    if curl -s --max-time 5 "$FASTAPI_BASE_URL/openapi.json" > /dev/null 2>&1; then
        log_success "FastAPI server is running"
        return 0
    else
        log_error "FastAPI server not responding at $FASTAPI_BASE_URL"
        log_info "Start it first: uvicorn app:app --port 8000"
        return 1
    fi
}

get_domain_port() {
    local domain=$1
    for i in "${!DOMAINS[@]}"; do
        if [ "${DOMAINS[$i]}" = "$domain" ]; then
            echo $((BASE_PORT + i))
            return 0
        fi
    done
    echo 0
}

start_one() {
    local domain=$1
    local port
    port=$(get_domain_port "$domain")
    local pid_file="$PID_DIR/${domain}.pid"
    local log_file="$LOG_DIR/${domain}.log"

    if [ "$port" -eq 0 ]; then
        log_error "Unknown domain: $domain"
        return 1
    fi

    # Already running?
    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            log_warning "${domain}-mcp already running (PID $pid, port $port)"
            return 0
        fi
        rm -f "$pid_file"
    fi

    FASTAPI_BASE_URL="$FASTAPI_BASE_URL" \
    MCP_SERVER_NAME="${domain}-mcp" \
    MCP_DOMAINS="$domain" \
    nohup python mcp_server.py > "$log_file" 2>&1 &

    local pid=$!
    echo "$pid" > "$pid_file"

    sleep 0.2
    if ps -p "$pid" > /dev/null 2>&1; then
        log_success "${domain}-mcp  (PID $pid, port $port)"
    else
        log_error "${domain}-mcp failed to start — see $log_file"
        rm -f "$pid_file"
        return 1
    fi
}

stop_one() {
    local domain=$1
    local pid_file="$PID_DIR/${domain}.pid"

    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            kill "$pid" 2>/dev/null || true
            sleep 0.3
            ps -p "$pid" > /dev/null 2>&1 && kill -9 "$pid" 2>/dev/null || true
            log_success "Stopped ${domain}-mcp (PID $pid)"
        fi
        rm -f "$pid_file"
    fi
}

status_one() {
    local domain=$1
    local port
    port=$(get_domain_port "$domain")
    local pid_file="$PID_DIR/${domain}.pid"

    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            printf "  ${GREEN}●${NC} %-35s port %-5s PID %s\n" "${domain}-mcp" "$port" "$pid"
            return 0
        else
            printf "  ${RED}●${NC} %-35s port %-5s (dead)\n" "${domain}-mcp" "$port"
            return 1
        fi
    else
        printf "  ${RED}○${NC} %-35s port %-5s (stopped)\n" "${domain}-mcp" "$port"
        return 1
    fi
}

# ---- aggregate commands ----

cmd_start() {
    local targets=("$@")
    [ ${#targets[@]} -eq 0 ] && targets=("${DOMAINS[@]}")

    echo ""
    echo "=========================================="
    echo "  Starting ${#targets[@]} MCP servers"
    echo "=========================================="
    echo ""

    check_fastapi || exit 1
    echo ""

    local failed=0
    for d in "${targets[@]}"; do
        start_one "$d" || ((failed++))
    done

    echo ""
    if [ $failed -eq 0 ]; then
        log_success "All ${#targets[@]} servers started"
    else
        log_warning "$failed server(s) failed"
    fi
    echo ""
}

cmd_stop() {
    echo ""
    echo "=========================================="
    echo "  Stopping all MCP servers"
    echo "=========================================="
    echo ""
    for d in "${DOMAINS[@]}"; do stop_one "$d"; done
    log_success "Done"
    echo ""
}

cmd_status() {
    echo ""
    echo "=========================================="
    echo "  MCP Server Status"
    echo "=========================================="
    echo ""

    local running=0 total=0
    for d in "${DOMAINS[@]}"; do
        ((total++))
        status_one "$d" && ((running++))
    done

    echo ""
    echo "  Running: $running / $total"
    echo ""
}

cmd_list() {
    echo ""
    echo "  #   Domain                              Port"
    echo "  --- ----------------------------------- -----"
    for i in "${!DOMAINS[@]}"; do
        printf "  %3d %-35s %d\n" "$((i+1))" "${DOMAINS[$i]}" "$((BASE_PORT+i))"
    done
    echo ""
    echo "  Total: ${#DOMAINS[@]} domains  (ports ${BASE_PORT}–$((BASE_PORT + ${#DOMAINS[@]} - 1)))"
    echo ""
}

usage() {
    cat <<EOF

Usage: $0 [command] [domains...]

Commands:
  start [domain...]   Start all 80 servers (or specific domains)
  stop                Stop all servers
  restart             Restart all servers
  status              Show running/stopped status
  list                Print all 80 domains with port assignments
  logs <domain>       Tail logs for a domain

Examples:
  $0                           # Start all 80
  $0 start hockey movie        # Start just these two
  $0 stop                      # Stop everything
  $0 status                    # What's running?
  $0 list                      # Show domain → port mapping
  $0 logs hockey               # Tail hockey-mcp logs

Environment:
  FASTAPI_BASE_URL   (default: http://localhost:8000)
  BASE_PORT          (default: 8001)
  LOG_DIR            (default: ./logs/mcp)
  PID_DIR            (default: ./pids)

EOF
}

# ---- main ----

setup_dirs

case "${1:-start}" in
    start)
        shift 2>/dev/null || true
        cmd_start "$@"
        ;;
    stop)       cmd_stop ;;
    restart)    cmd_stop; sleep 2; cmd_start ;;
    status)     cmd_status ;;
    list)       cmd_list ;;
    logs)
        [ -z "$2" ] && { log_error "Specify a domain: $0 logs hockey"; exit 1; }
        f="$LOG_DIR/${2}.log"
        [ -f "$f" ] && tail -f "$f" || { log_error "No log file for '$2'"; exit 1; }
        ;;
    -h|--help|help) usage ;;
    *) log_error "Unknown command: $1"; usage; exit 1 ;;
esac
