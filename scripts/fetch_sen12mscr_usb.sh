#!/usr/bin/env bash
# Download SEN12MS-CR archives from the TUM data server straight onto a pendrive (macOS/Linux).
#
#   bash scripts/fetch_sen12mscr_usb.sh /Volumes/DATA/sen12ms_cr                # spring only (~70 GB)
#   bash scripts/fetch_sen12mscr_usb.sh /Volumes/DATA/sen12ms_cr spring summer  # more seasons
#   bash scripts/fetch_sen12mscr_usb.sh --dry-run /Volumes/DATA/sen12ms_cr      # sizes + space check only
#
# The rsync password is on https://mediatum.ub.tum.de/1554803. Set RSYNC_PASSWORD or you will be
# asked once; it is never written to disk. Safe to re-run: finished files are skipped, partial ones
# resume. Every archive is checked against the official checksums.sha512 at the end.
# The pendrive must be exFAT (FAT32 cannot hold files over 4 GB).
set -euo pipefail

SERVER="rsync://m1554803@dataserv.ub.tum.de/m1554803"
MAX_TRIES=20

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN=1; shift; fi
if [ $# -lt 1 ]; then
  echo "usage: $0 [--dry-run] DEST_DIR [spring|summer|fall|winter ...]" >&2
  exit 1
fi
DEST="$1"; shift
SEASONS=("$@")
[ ${#SEASONS[@]} -eq 0 ] && SEASONS=(spring)

roi() {
  case "$1" in
    spring) echo ROIs1158_spring ;;
    summer) echo ROIs1868_summer ;;
    fall)   echo ROIs1970_fall ;;
    winter) echo ROIs2017_winter ;;
    *) echo "unknown season: $1 (use spring, summer, fall or winter)" >&2; exit 1 ;;
  esac
}

FILES=()
for s in "${SEASONS[@]}"; do
  r=$(roi "$s")
  FILES+=("${r}_s1.tar.gz" "${r}_s2.tar.gz" "${r}_s2_cloudy.tar.gz")
done

# --- destination checks -------------------------------------------------------------------------
PARENT=$(dirname "$DEST")
if [ ! -d "$PARENT" ]; then
  echo "error: $PARENT does not exist. Is the pendrive plugged in? (ls /Volumes)" >&2
  exit 1
fi
mkdir -p "$DEST"
cd "$DEST"

MOUNT=$(df -P . | awk 'NR==2 {print $6}')
FSTYPE=$(mount | awk -v m="$MOUNT" '$3 == m {print $4}' | tr -d '(,')
echo "destination: $DEST  (mount $MOUNT, filesystem ${FSTYPE:-unknown})"
case "$FSTYPE" in
  msdos|vfat|fat32)
    echo "error: the pendrive is FAT32, which cannot hold files over 4 GB. Reformat it as exFAT." >&2
    exit 1 ;;
esac

if [ -z "${RSYNC_PASSWORD:-}" ]; then
  read -r -s -p "rsync password (from the mediaTUM page): " RSYNC_PASSWORD; echo
  export RSYNC_PASSWORD
fi

# --- sizes and free space -----------------------------------------------------------------------
LISTING=$(rsync --list-only "$SERVER/")
NEED=0
for f in "${FILES[@]}"; do
  size=$(echo "$LISTING" | awk -v f="$f" '$NF == f {print $2}')
  if [ -z "$size" ]; then echo "error: $f not found on the server" >&2; exit 1; fi
  have=0
  [ -f "$f" ] && have=$(wc -c < "$f" | tr -d ' ')
  NEED=$((NEED + size - have))
  printf '  %-36s %6.1f GB  (have %.1f GB)\n' "$f" "$(echo "$size / 1e9" | bc -l)" "$(echo "$have / 1e9" | bc -l)"
done
FREE=$(( $(df -Pk . | awk 'NR==2 {print $4}') * 1024 ))
printf 'still to download: %.1f GB, free on pendrive: %.1f GB\n' "$(echo "$NEED / 1e9" | bc -l)" "$(echo "$FREE / 1e9" | bc -l)"
if [ "$NEED" -gt "$FREE" ]; then
  echo "error: not enough space on the pendrive" >&2
  exit 1
fi
if [ "$DRY_RUN" = 1 ]; then echo "dry run: nothing downloaded"; exit 0; fi

# --- download (keeps the Mac awake on macOS) ----------------------------------------------------
KEEP_AWAKE=()
command -v caffeinate >/dev/null && KEEP_AWAKE=(caffeinate -i)

rsync -q "$SERVER/checksums.sha512" .
for f in "${FILES[@]}"; do
  try=1
  # --append resumes a partial file instead of re-reading it; the checksum step below catches corruption
  until ${KEEP_AWAKE[@]+"${KEEP_AWAKE[@]}"} rsync -t --partial --append --progress --timeout=300 "$SERVER/$f" .; do
    if [ $try -ge $MAX_TRIES ]; then echo "error: $f failed after $MAX_TRIES tries" >&2; exit 1; fi
    echo "connection dropped on $f (try $try/$MAX_TRIES), retrying in 30 s..."
    try=$((try + 1)); sleep 30
  done
done

# --- verify -------------------------------------------------------------------------------------
echo "verifying checksums (reads every file once, takes a while)..."
FAILED=0
for f in "${FILES[@]}"; do
  if grep " \./$f\$" checksums.sha512 | shasum -a 512 -c --status; then
    echo "  OK   $f"
  else
    echo "  BAD  $f  -> delete it and re-run this script"; FAILED=1
  fi
done
[ "$FAILED" = 0 ] || exit 1
echo "done. Eject with: diskutil eject \"$MOUNT\""
